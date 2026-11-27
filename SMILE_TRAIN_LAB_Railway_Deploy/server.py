"""SMILE TRAIN LAB public server. Isolates each visitor's camera and renderer."""
from __future__ import annotations
import hashlib, json, multiprocessing as mp, os, queue, re, secrets, sqlite3, threading, time, urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_ACTIVE = max(1, int(os.environ.get('MAX_ACTIVE_SESSIONS', '3')))
SESSION_TTL = 3600
MAX_CARD_SIZE = 8_000_000
MAX_ARCHIVED_CARDS = 600
MAX_ARCHIVED_BYTES = 1_000_000_000
DBDIR = Path(os.environ.get('SMILE_DATA_DIR', os.environ.get('RAILWAY_VOLUME_MOUNT_PATH', str(ROOT / 'local-data'))))
DBDIR.mkdir(parents=True, exist_ok=True)
DB = DBDIR / 'archive.sqlite3'
archive_lock = threading.Lock()
state_lock = threading.RLock()
sessions = {}
context = mp.get_context('spawn')


def db():
    con = sqlite3.connect(DB, timeout=15)
    con.execute('PRAGMA busy_timeout=15000')
    con.execute('CREATE TABLE IF NOT EXISTS cards (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, image BLOB NOT NULL, created REAL NOT NULL)')
    return con


def worker_main(camera_q, key_q, frame_q, event_q):
    """Every visitor gets a new Python process and isolated OpenCV/MediaPipe state."""
    try:
        import cv2
        import numpy as np
        import traceback
        cv2.setNumThreads(1)
        os.chdir(ROOT)
        source = (ROOT / 'original.py').read_text(encoding='utf-8')
        source = source.replace('cap = find_working_camera()', 'cap = Camera()')
        source = source.replace("os.system(f\"lpr '{card_filename}' &\")", 'pass')
        # The macOS audio command is not present on Linux; omit the sound, not the UI.
        source = re.sub(r"subprocess\.Popen\(\['afplay'.*?\)\s*$", 'pass', source, flags=re.M)
        source = source.replace("os.path.expanduser('~/Desktop/face_landmarker.task')", "str(ROOT / 'face_landmarker.task')")
        source = source.replace('"/Library/Fonts/Arial Unicode.ttf"', '"/Library/Fonts/Arial Unicode.ttf", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"')

        class Camera:
            def __init__(self): self.last = None
            def read(self):
                try:
                    while True:
                        b = camera_q.get_nowait()
                        latest = b
                except queue.Empty: pass
                if 'latest' in locals():
                    decoded = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
                    if decoded is not None: self.last = decoded
                return self.last is not None, self.last.copy() if self.last is not None else None
            def release(self): pass
            def isOpened(self): return True

        last_image = None
        last_encoded = None
        def send_frame(name, img):
            nonlocal last_image, last_encoded
            if last_image is not None and np.array_equal(img, last_image):
                return
            last_image = img.copy()
            static = bool(np.mean(np.all(img[::16, ::16] == img[0, 0], axis=2)) > .55)
            ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 1]) if static else cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if ok:
                try: frame_q.put_nowait(buf.tobytes())
                except queue.Full:
                    try: frame_q.get_nowait()
                    except queue.Empty: pass
                    try: frame_q.put_nowait(buf.tobytes())
                    except queue.Full: pass

        def wait_key(ms):
            try: return key_q.get(timeout=max(0.001, min(ms / 1000.0, 0.1)))
            except queue.Empty: return -1

        def save_card(path, img):
            ok, buf = cv2.imencode('.png', img)
            if ok: event_q.put(('card', buf.tobytes()))
            return bool(ok)

        cv2.imshow = send_frame
        cv2.waitKey = wait_key
        cv2.waitKeyEx = wait_key
        cv2.namedWindow = lambda *args, **kwargs: None
        cv2.resizeWindow = lambda *args, **kwargs: None
        cv2.destroyAllWindows = lambda *args, **kwargs: None
        cv2.imwrite = save_card
        exec(compile(source, str(ROOT / 'original.py'), 'exec'), {'Camera': Camera, 'ROOT': ROOT, '__name__': '__main__'})
    except BaseException as exc:
        import traceback
        try: event_q.put(('error', str(exc)[:700]))
        except Exception: pass
        traceback.print_exc()


class Session:
    def __init__(self):
        self.token = secrets.token_urlsafe(32)
        self.created = self.accessed = time.time()
        self.camera_q = context.Queue(maxsize=2)
        self.key_q = context.Queue(maxsize=64)
        self.frame_q = context.Queue(maxsize=2)
        self.event_q = context.Queue(maxsize=4)
        self.proc = None
        self.frame = None
        self.card = None
        self.error = ''
        self.published = False
        self.mutex = threading.Lock()

    def poll(self):
        with self.mutex:
            for _ in range(4):
                try: self.frame = self.frame_q.get_nowait()
                except queue.Empty: break
            for _ in range(5):
                try: kind, data = self.event_q.get_nowait()
                except queue.Empty: break
                if kind == 'card': self.card = data
                if kind == 'error': self.error = data
            if self.proc and not self.proc.is_alive() and self.proc.exitcode not in (0, None) and not self.error:
                self.error = '얼굴 분석 프로세스가 종료되었습니다. 다시 시도해 주세요.'

    def stop(self):
        if self.proc and self.proc.is_alive():
            try: self.key_q.put_nowait(27)
            except queue.Full: pass
            self.proc.join(timeout=.25)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=.5)
        self.proc = None


def cleanup_sessions():
    now = time.time()
    with state_lock:
        for k, session in list(sessions.items()):
            if now - session.accessed > SESSION_TTL:
                session.stop()
                del sessions[k]


class Handler(SimpleHTTPRequestHandler):
    server_version = 'SmileTrainPublic/1.0'

    def log_message(self, format, *args):
        # Never log query strings or identifying session tokens.
        return

    def reply(self, data=b'', kind='text/plain; charset=utf-8', status=200, head=False):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store' if self.path.startswith('/api/') else 'public, max-age=1800')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        if not head:
            try: self.wfile.write(data)
            except (ConnectionAbortedError, BrokenPipeError): pass

    def current(self):
        token = self.headers.get('X-Session-Token', '')
        with state_lock:
            session = sessions.get(token)
            if session: session.accessed = time.time()
        return session

    def protected(self):
        session = self.current()
        if not session: self.reply(b'Invalid or expired session', status=401)
        return session

    def handle_static(self, head=False):
        path = urllib.parse.urlsplit(self.path).path
        if path in ('/', '/index.html'):
            file = ROOT / 'index.html'
        elif path in ('/app.js', '/gallery.js'):
            file = ROOT / path.lstrip('/')
        elif re.fullmatch(r'/assets/[a-zA-Z0-9._-]+', path):
            file = ROOT / path.lstrip('/')
        else:
            self.reply(b'Not found', status=404, head=head)
            return
        if not file.is_file():
            self.reply(b'Not found', status=404, head=head)
            return
        ext = file.suffix.lower()
        kind = {'.html':'text/html; charset=utf-8', '.js':'text/javascript; charset=utf-8', '.jpg':'image/jpeg', '.png':'image/png', '.bin':'application/octet-stream'}.get(ext,'application/octet-stream')
        self.reply(file.read_bytes(),kind,head=head)

    def do_HEAD(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/api/card':
            s = self.current()
            if not s: self.reply(status=401,head=True); return
            s.poll()
            self.reply(s.card or b'',kind='image/png',status=200 if s.card else 204,head=True)
        else: self.handle_static(head=True)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == '/health':
            self.reply(b'ok');return
        if path == '/api/archive':
            with archive_lock, db() as c:
                rows = c.execute('SELECT id FROM cards ORDER BY created DESC LIMIT 120').fetchall()
            self.reply(json.dumps([{'id':r[0]} for r in rows]).encode(),'application/json');return
        m = re.fullmatch(r'/api/archive/([0-9a-f]{32})', path)
        if m:
            with archive_lock, db() as c:
                row = c.execute('SELECT image FROM cards WHERE id=?', (m[1],)).fetchone()
            self.reply(row[0] if row else b'','image/png',200 if row else 404); return
        if path in ('/api/frame', '/api/card', '/api/status'):
            s = self.protected()
            if not s:return
            s.poll()
            if path == '/api/frame': self.reply(s.frame or b'','image/png' if s.frame and s.frame.startswith(b'\x89PNG') else 'image/jpeg',200 if s.frame else 204)
            elif path == '/api/card': self.reply(s.card or b'','image/png',200 if s.card else 204)
            else: self.reply(s.error.encode())
            return
        self.handle_static()

    def do_POST(self):
        cleanup_sessions()
        path = urllib.parse.urlsplit(self.path).path
        try: length = int(self.headers.get('Content-Length', '0'))
        except ValueError: length = -1
        if length < 0 or length > MAX_CARD_SIZE:
            self.reply(b'Invalid size',status=413);return
        if path == '/api/session':
            with state_lock:
                if len(sessions) >= 100:
                    self.reply(b'Too many sessions, try again later',status=503);return
                s = Session(); sessions[s.token] = s
            self.reply(json.dumps({'token':s.token}).encode(),'application/json');return
        s = self.protected()
        if not s:return
        data = self.rfile.read(length)
        if path == '/api/local-card':
            import struct
            if s.card or s.published or not (1000 <= length <= MAX_CARD_SIZE) or not data.startswith(b'\x89PNG\r\n\x1a\n') or data[12:16] != b'IHDR' or struct.unpack('>II',data[16:24]) != (1000,1480):
                self.reply(b'Invalid local result card',status=400);return
            s.card=data
            self.reply();return
        if path == '/api/camera':
            if not s.proc or not s.proc.is_alive():
                self.reply(b'Session is not active',status=409);return
            if length < 10 or length > 800_000:
                self.reply(b'Invalid camera frame size',status=413);return
            try: s.camera_q.put_nowait(data)
            except queue.Full:
                try:s.camera_q.get_nowait()
                except queue.Empty:pass
                try:s.camera_q.put_nowait(data)
                except queue.Full:pass
            self.reply();return
        if path == '/api/start':
            with state_lock:
                if s.proc and s.proc.is_alive():
                    self.reply(b'Already running',status=409);return
                running = sum(1 for v in sessions.values() if v.proc and v.proc.is_alive())
                if running >= MAX_ACTIVE:
                    self.reply('동시 체험 인원이 많습니다. 잠시 후 다시 시도해 주세요.'.encode(),status=503);return
                s.frame=s.card=None;s.error='';s.published=False
                s.proc=context.Process(target=worker_main,args=(s.camera_q,s.key_q,s.frame_q,s.event_q),daemon=True)
                s.proc.start()
            self.reply();return
        if path == '/api/key':
            if not s.proc or not s.proc.is_alive():self.reply(b'Not running',status=409);return
            try:
                value=int(data)
                if not (-1 <= value <= 2_000_000):raise ValueError
                s.key_q.put_nowait(value)
            except (ValueError,queue.Full):self.reply(b'Invalid key',status=400);return
            self.reply();return
        if path == '/api/stop':
            s.stop();self.reply();return
        if path == '/api/archive':
            s.poll()
            if not s.card or data != s.card or s.published:
                self.reply(b'Only the current unpublished result card may be shared',status=403);return
            if not (1000 <= length <= MAX_CARD_SIZE) or not data.startswith(b'\x89PNG\r\n\x1a\n'):
                self.reply(b'Invalid PNG',status=400);return
            ident = secrets.token_hex(16);delete_token=secrets.token_urlsafe(32)
            with archive_lock, db() as c:
                count, total_bytes = c.execute('SELECT COUNT(*), COALESCE(SUM(LENGTH(image)),0) FROM cards').fetchone()
                if count >= MAX_ARCHIVED_CARDS or total_bytes + len(data) > MAX_ARCHIVED_BYTES:
                    self.reply('공동 전시 보관 공간이 가득 찼습니다.'.encode(),status=507); return
                c.execute('INSERT INTO cards VALUES(?,?,?,?)',(ident,hashlib.sha256(delete_token.encode()).hexdigest(),data,time.time()))
            s.published=True
            self.reply(json.dumps({'id':ident,'token':delete_token}).encode(),'application/json');return
        self.reply(b'Not found',status=404)

    def do_DELETE(self):
        path=urllib.parse.urlsplit(self.path).path
        m=re.fullmatch(r'/api/archive/([0-9a-f]{32})',path)
        if not m:self.reply(status=404);return
        token=self.headers.get('X-Delete-Token','')
        if len(token)>150:self.reply(status=403);return
        with archive_lock,db() as c:
            cur=c.execute('DELETE FROM cards WHERE id=? AND token_hash=?',(m[1],hashlib.sha256(token.encode()).hexdigest()))
        self.reply(status=200 if cur.rowcount else 403)


if __name__=='__main__':
    db().close()
    host='0.0.0.0'
    port=int(os.environ.get('PORT','8000'))
    print(f'SMILE TRAIN LAB public server listening on {host}:{port}',flush=True)
    ThreadingHTTPServer((host,port),Handler).serve_forever()
