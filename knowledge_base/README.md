# Knowledge Base - DOTA Aerial Object Categories

## DEMO / SYNTHETIC KNOWLEDGE BASE

> **IMPORTANT NOTICE**: This document and the associated `base_knowledge.json` file contain **synthetic/demonstrative reference information** created specifically for an academic college project demonstration. They are **NOT** intended to represent authoritative real-world operational intelligence.

---

### Purpose
The purpose of this knowledge base is to demonstrate a Retrieval-Augmented Generation (RAG) architecture integrated with aerial image object detection.

When an aerial object category (e.g. `plane`, `ship`, `storage-tank`, etc.) is detected by the computer vision model, the system queries this knowledge base to retrieve category-specific technical and context information, which is then fed into the LLM prompt.

---

### Covered Categories (15 Canonical DOTA Classes)
1. **plane** - Fixed-wing aircraft
2. **ship** - Marine vessels
3. **storage-tank** - Industrial liquid/fuel storage cylinders
4. **baseball-diamond** - Baseball sports field layout
5. **tennis-court** - Rectangular tennis court
6. **basketball-court** - Outdoor basketball court
7. **ground-track-field** - Oval running track field
8. **harbor** - Port/pier facilities
9. **bridge** - Elevated transportation superstructure
10. **large-vehicle** - Freight trucks, buses, trailers
11. **small-vehicle** - Passenger cars, sedans, SUVs
12. **helicopter** - Rotary-wing aircraft
13. **roundabout** - Circular road junction
14. **soccer-ball-field** - Soccer athletic field
15. **swimming-pool** - Water basin structure
