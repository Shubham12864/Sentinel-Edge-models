# Sentinel Edge Model

This repository is the Tier 2 AI model and demo code for a face-recognition surveillance pipeline.

It is not a full end-user web app. It is a model architecture and working local demo that shows how the system is designed to work in real deployment.

## What this repository is

This project takes a camera frame, detects faces, tracks them, extracts embeddings, compares them with known identities, and emits a result event.

The flow is:

- Tier 1: camera or source provides frames
- Tier 2: this model pipeline processes frames
- Tier 3: downstream app or dashboard consumes events

This repo focuses on Tier 2.

## Main idea

The model works like this:

1. A frame comes in from a camera
2. YOLO detects faces
3. ByteTrack keeps track of each face over time
4. A face crop is extracted
5. ArcFace creates a 512-dimensional embedding
6. FAISS searches the known identity database
7. The system decides whether the person is verified
8. An event is returned with identity, score, camera, and timestamp

## Folder structure

```text
Sentinel Edge models/
├── yolo26n.pt
├── README.md
├── .gitignore
├── tier2-ai/
│   ├── README.md
│   ├── requirements.txt
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── notebooks/
│   ├── sample_data/
│   ├── scripts/
│   ├── src/
│   └── tests/
├── runs/
└── ...
```

## Important note

This repository is a model architecture and demo layer, not a complete front-end product.

It does not include:

- a full website
- a complete camera server
- a full dashboard UI
- a complete Tier 3 app

Instead, it gives the AI core and a local camera-based validation flow.

## Local demo

Use the demo script inside the Tier 2 project:

```bash
cd tier2-ai
python tests/demo.py
```

This test opens the laptop camera, sends frames through the pipeline, and prints event output in the terminal.

## Identity enrollment

Put your known face images in:

```text
tier2-ai/data/known_identities/
```

Then run:

```bash
tier2-ai\scripts\enroll_identities.py
```

or from the project root:

```bash
cd tier2-ai
python scripts/enroll_identities.py
```

This builds the FAISS identity index from the folder images.

## Real pipeline flow

```text
Camera / Source
    ↓
Tier 1: FramePacket creation
    ↓
Tier 2: Face detection + tracking + embedding + FAISS match
    ↓
Tier 3: Event receiver / alert dashboard / downstream app
```

## Main project details

The model code is under:

- [tier2-ai/src](tier2-ai/src)
- [tier2-ai/scripts](tier2-ai/scripts)
- [tier2-ai/tests](tier2-ai/tests)

The main processing entry point is:

- `UnifiedPipeline.process_frame_packet(...)`

This is the place where a frame becomes an AI event.

## Requirements

The project dependencies are listed in:

- [tier2-ai/requirements.txt](tier2-ai/requirements.txt)

Install them with:

```bash
cd tier2-ai
pip install -r requirements.txt
```

## Model files

The YOLO weights are expected in:

```text
tier2-ai/models/yolo26n/yolo26 widerdataset.pt
```

## Practical use

This is suitable for:

- local testing
- model validation
- edge AI experiments
- future cloud deployment
- building upstream and downstream integrations

It is not a complete final product by itself.

## License

This project is shared for learning and model experimentation. Add your own license if you plan to publish it publicly.

