# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Counter-Drone C2 Simulator**

A small-unit low-cost counter-drone command & control system simulator.

## Build & Run Commands

### Simulator Server (Node.js + TypeScript)
```bash
cd simulator
npm install
npm run dev          # Dev server at ws://localhost:8080
npm run build        # TypeScript compilation
npm run test         # Run all Jest tests
npm run test:watch   # Watch mode for tests
```

### Frontend (React + TypeScript)
```bash
cd frontend
npm install
npm run dev          # Dev server at http://localhost:3000
npm run build        # Production build (tsc + vite)
```

### Batch Experiments & Evaluation
```bash
cd simulator
npm run batch:quick      # 5 runs, 30s each
npm run batch:full       # 50 runs, 120s each
npm run eval:fast        # Fast evaluation profile
npm run eval:full        # Full evaluation (paper-quality)
```

### Python Analysis
```bash
cd analysis
pip install -r requirements.txt
python scripts/generate_report.py --full
python auto_tune.py --trials 30 --profile fast
```

## Architecture Overview

### Layer Structure
```
WebSocket Layer (ws://localhost:8080)
    ↓
Simulation Engine (simulation.ts - tick-based, 100ms interval)
    ↓
Core Modules:
├── Sensor Fusion (core/fusion/) - EKF/weighted avg modes
├── Threat Assessment (core/fusion/threatScore.ts) - multi-factor scoring
├── Engagement Manager (core/engagement/) - intercept decisions
└── Logging (core/logging/) - JSONL event logging
    ↓
Sensor & Behavior Models:
├── Sensors: radar.ts, acousticSensor.ts, eoSensor.ts
└── Models: hostileDrone.ts, interceptor.ts, guidance.ts
    ↓
Adapters (adapters/) - Factory pattern for 2D internal sensors/controllers
```

### Communication Flow
- Frontend connects via WebSocket to simulator server
- Bidirectional JSON messages validated with Zod schemas
- All events logged to `simulator/logs/*.jsonl` for analysis

### Key Design Patterns
- **Factory Pattern**: `AdapterFactory` for sensor/drone controller creation
- **State Machine**: Interceptor drone states (STANDBY → LAUNCHING → PURSUING → ENGAGING → RETURNING)
- **Adapter Pattern**: Abstraction layer for different simulation modes (currently 2D-only)

## Key Source Files

| File | Purpose |
|------|---------|
| `simulator/src/simulation.ts` | Main simulation engine (~50K lines) |
| `simulator/src/core/fusion/sensorFusion.ts` | Multi-sensor fusion logic |
| `simulator/src/core/fusion/threatScore.ts` | Threat assessment (0-100 score) |
| `simulator/src/models/interceptor.ts` | Interceptor state machine + PN guidance |
| `simulator/src/models/hostileDrone.ts` | Enemy drone behavior modes |
| `shared/schemas.ts` | Zod schemas for WebSocket messages |
| `frontend/src/hooks/useWebSocket.ts` | WebSocket connection management |

## Coding Conventions

### TypeScript
- Strict mode enabled in both simulator and frontend
- Types in `types/` directories or `types.ts` files
- Shared types in `shared/schemas.ts` (Zod schemas → TypeScript types)
- Path alias: `@shared/*` → `../shared/*` (simulator only)

### Testing
- Jest with ts-jest preset
- Tests in `simulator/src/__tests__/*.test.ts`
- Run single test: `npm run test -- --testPathPattern="threatScore"`

### Logging
- JSONL format in `simulator/logs/`
- Event types: drone_spawned, radar_detection, audio_detection, eo_detection, fused_track_update, threat_score_update, engage_command, intercept_result, manual_action

## Domain Concepts

### Threat Level Thresholds
- CRITICAL: 75+ (immediate response)
- DANGER: 50-74 (prepare response)
- CAUTION: 25-49 (monitor)
- INFO: 0-24 (information gathering)

### Drone Behavior Modes
- **Enemy**: NORMAL, RECON, ATTACK_RUN, EVADE
- **Interceptor**: STANDBY, LAUNCHING, PURSUING, ENGAGING, RETURNING

### Intercept Methods
- RAM (collision, <5m), GUN (ballistic, 100-400m), NET (net gun, <100m), JAM (jamming, 50-300m)

## Environment Configuration

Copy `simulator/.env.example` to `.env`:
- `SIMULATOR_PORT`: WebSocket port (default: 8080)
- `LOG_ENABLED`: Enable JSONL logging
- `SIM_MODE`: INTERNAL (2D-only mode)
