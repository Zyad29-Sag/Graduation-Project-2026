"""
modules/face
------------
Part 11 — additive face-side features merged from the team branch:
  - FaceAnalyzer       : InsightFace detection/recognition + age/gender +
                         named watchlist + "returning face" badge.
  - EthnicityClassifier: optional ResNet18 ethnicity head (graceful-disable).
  - FaceSearcher       : face-image search over the ISOLATED face_embeddings
                         table (never touches body identity / person_embeddings).

Design rule: nothing in this package may influence cross-camera BODY identity.
Face contributes display attributes, watchlist names, a "returning" badge, and
a dedicated face-image search path only.
"""
