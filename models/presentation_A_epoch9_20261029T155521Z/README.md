# Separate supervised A presentation export

Official selected expanded supervised epoch9, inference-only parameters and all BN buffers. Original checkpoint and published epoch4 weights/defaults are unchanged.

Architecture, preprocessing, strict-load/parity checks and parent/export hashes are in supervised_A_epoch9.json. Full validation A trade-off:11.84% lower pooled AbsRel but16.18% higher near RMSE versus original. This is a general-depth demonstration candidate, not a near-field safety improvement.

Use the optional local dashboard documented in docs/presentation_dashboard.md. No teacher is required for student inference. See project NOTICE.md for upstream model/dataset terms; no new blanket licence or deployment claim is introduced.
