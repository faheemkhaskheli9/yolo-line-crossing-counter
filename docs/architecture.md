# Architecture Notes: Object Line-Crossing Counter

## Pipeline

```text
Video -> Detection + Tracking -> Line-Crossing Logic -> In/Out Counters -> Overlay Output
```

## Components

- Object detection
- Tracking
- Configurable counting line
- Direction detection (in/out)
- Incoming count
- Outgoing count
- Live visualization overlay

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
