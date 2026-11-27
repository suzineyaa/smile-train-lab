# SMILE TRAIN LAB — Railway deployment

Contains the V15 3D exhibition and original Python UI rendering, migrated for multi-user public access. No re-design of the exhibition is intended. Each visitor uses a dedicated camera-processing worker.

## Publish
1. Create a **private** GitHub repository, and upload **the contents** of this folder to its root.
2. Connect the repository to the existing Railway `smile-train-lab` project. Railway uses `Dockerfile` and `railway.json` to build and run the service.
3. Attach a Railway persistent volume to the application service, with mount path `/data`. It stores only the voluntarily published result cards and deletion tokens.
4. Generate a Railway public domain and verify `https://<domain>/health` responds `ok`.
5. On two devices, test permissions, camera capture, result-card download, consent-based archive publication, display on the shelf, and removal.

## Privacy
A camera frame is securely uploaded to the server for real-time facial analysis, held temporarily in memory, then discarded; it is not saved to disk by the server. A result image is publicly archived **only after explicit consent**. Users can remove their own card using a secret deletion token stored in their browser localStorage, but if they clear browser data they will need assistance. Published cards include the user's name and face, and remain until manually deleted.

## Limits / costs
The default cap is three concurrent running training sessions, subject to the actual Railway CPU and memory capacity. Railway service and volume storage may incur charges. When multiple instances or replicas are used, SQLite on a single volume is insufficient: use a shared database and object storage instead. Keep the web process to one replica.

## Differences from macOS
The native Apple font is not bundled for licensing reasons; Noto Sans CJK is substituted in the Linux container, which may slightly change text metrics. Mac system audio effects and physical printing are disabled; card download and virtual printing remain.
