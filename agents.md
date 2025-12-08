# FracFire AI Agent Integration - Future Improvements & Cleanup

## Overview

This document tracks the integration of the fracfireagent (AI-powered GUI) with the FracFire synthetic data generation platform. It outlines completed work, current status, and future refactoring/improvement ideas.

## Completed Integration Work

### 1. Repository Restructuring
- ✅ Converted to monorepo structure
- ✅ Moved fracfireagent to `frontend/` directory
- ✅ Created root `package.json` with workspace configuration
- ✅ Set up unified build scripts (`npm run dev`, `npm install`)

### 2. Backend API (FastAPI)
- ✅ Created `api/` directory with FastAPI backend
- ✅ Exposed price generator as REST endpoints
- ✅ Integrated real `PriceGenerator` from `src/core/generator/`
- ✅ Added dataset management endpoints (list, get, upload)
- ✅ Added model training endpoints (mock implementation)
- ✅ Added setup analysis endpoints (mock implementation)
- ✅ CORS configuration for local development

### 3. Frontend Updates
- ✅ Created `api-client.ts` for backend communication
- ✅ Created `RealBackend` service to replace `MockBackend`
- ✅ Updated React app to use real API calls
- ✅ Added CSS styling
- ✅ Updated README with installation instructions

## Current Architecture

```
fracfire/
├── api/                    # FastAPI Backend
│   ├── main.py            # Main API server with endpoints
│   ├── requirements.txt   # API-specific dependencies
│   └── .env.example       # Environment variables template
├── frontend/              # React Frontend (formerly fracfireagent)
│   ├── index.tsx          # Main React application
│   ├── api-client.ts      # API communication layer
│   ├── backend-service.ts # Backend service abstraction
│   ├── index.html         # HTML template
│   ├── index.css          # Global styles
│   ├── vite.config.ts     # Vite configuration
│   └── package.json       # Frontend dependencies
├── src/                   # Python source code
│   ├── core/              # Core generators and engines
│   ├── ml/                # Machine learning models
│   └── agent/             # Agent tools
├── scripts/               # Python scripts for batch operations
├── lab/                   # Experimental code
├── data/                  # Data storage
│   ├── generated/         # Generated datasets
│   └── uploads/           # Uploaded datasets
├── models/                # Trained models
├── package.json           # Root package.json (monorepo)
├── requirements.txt       # Python dependencies
└── README.md              # Updated documentation
```

## Future Refactoring & Improvement Ideas

### High Priority

#### 1. Complete Real Backend Integration
**Current State:** Basic endpoints exist but use mock data for some operations  
**Todo:**
- [ ] Replace mock model training with real PyTorch/TensorFlow integration
- [ ] Replace mock setup analysis with real detector algorithms from `src/core/detector/`
- [ ] Implement real CNN training workflow using `src/ml/models/cnn.py`
- [ ] Add progress tracking for long-running operations (WebSocket or Server-Sent Events)
- [ ] Implement proper error handling and validation

#### 2. Setup Detection Integration
**Current State:** Endpoints exist but return mock data  
**Todo:**
- [ ] Connect to `src/core/detector/engine.py` for real setup detection
- [ ] Expose all detection styles from `src/core/detector/styles.py`
- [ ] Add configuration UI for detection parameters
- [ ] Implement setup library management from `src/core/detector/library.py`
- [ ] Add sweep functionality from `src/core/detector/sweep.py`

#### 3. Script Execution Framework
**Current State:** Basic script listing exists  
**Todo:**
- [ ] Create safe script execution sandbox
- [ ] Add script parameter UI forms
- [ ] Implement real-time output streaming
- [ ] Add script scheduling and batch execution
- [ ] Create script templates for common workflows

#### 4. Data Management Improvements
**Current State:** Basic upload/download works  
**Todo:**
- [ ] Add dataset versioning
- [ ] Implement data validation and quality checks
- [ ] Add dataset merging capabilities (as mentioned in requirements)
- [ ] Create data transformation pipeline UI
- [ ] Add export to multiple formats (CSV, Parquet, HDF5)

### Medium Priority

#### 5. Model Management System
**Todo:**
- [ ] Model versioning and tracking
- [ ] Hyperparameter tuning UI
- [ ] Model comparison dashboard
- [ ] Export/import trained models
- [ ] Model performance metrics visualization

#### 6. Trade Setup Lab Enhancements
**Todo:**
- [ ] Add more drawing tools (trendlines, zones)
- [ ] Implement setup templates library
- [ ] Add setup validation against historical data
- [ ] Create setup performance backtesting
- [ ] Add setup export to trading platforms

#### 7. Charting Improvements
**Todo:**
- [ ] Add technical indicators overlay
- [ ] Implement volume profile
- [ ] Add drawing persistence (save/load)
- [ ] Support multiple chart layouts
- [ ] Add chart snapshots/sharing

#### 8. AI Assistant Enhancements
**Todo:**
- [ ] Add more tool functions for the AI
- [ ] Implement conversation history
- [ ] Add suggested workflows/presets
- [ ] Create natural language query for data
- [ ] Add code generation for custom strategies

### Low Priority / Nice to Have

#### 9. User Experience
**Todo:**
- [ ] Add keyboard shortcuts
- [ ] Implement dark/light theme toggle
- [ ] Add customizable dashboard widgets
- [ ] Create onboarding tutorial
- [ ] Add context-sensitive help

#### 10. Performance Optimization
**Todo:**
- [ ] Implement data streaming for large datasets
- [ ] Add client-side caching
- [ ] Optimize chart rendering for large datasets
- [ ] Add Web Workers for heavy computations
- [ ] Implement progressive data loading

#### 11. Collaboration Features
**Todo:**
- [ ] Add user authentication
- [ ] Implement workspace sharing
- [ ] Add comments on charts and setups
- [ ] Create export/import for full workspaces
- [ ] Add version control for strategies

#### 12. Testing & Quality
**Todo:**
- [ ] Add unit tests for API endpoints
- [ ] Add integration tests for workflows
- [ ] Add E2E tests for frontend
- [ ] Implement API documentation (OpenAPI/Swagger)
- [ ] Add logging and monitoring

## Known Issues & Technical Debt

### 1. Generator Configuration
**Issue:** Current generator uses hardcoded parameters  
**Solution:** Expose all `PhysicsConfig`, `StateConfig`, etc. through API and UI

### 2. Data Persistence
**Issue:** No database, all data in JSON files  
**Solution:** Consider adding PostgreSQL or MongoDB for structured data storage

### 3. Model Training
**Issue:** Training blocks the API server  
**Solution:** Implement background job queue (Celery, RQ, or similar)

### 4. Frontend State Management
**Issue:** Complex state logic in main component  
**Solution:** Consider Redux, Zustand, or React Query for better state management

### 5. Type Safety
**Issue:** Types not shared between backend and frontend  
**Solution:** Generate TypeScript types from Python models or use OpenAPI code generation

### 6. Error Handling
**Issue:** Limited error feedback to user  
**Solution:** Implement comprehensive error handling with user-friendly messages

### 7. Authentication & Security
**Issue:** No authentication, API is open  
**Solution:** Add JWT authentication and API keys for production use

## Suggested Development Workflow

### For Adding New Features:

1. **Backend First:**
   - Add endpoint to `api/main.py`
   - Test with curl or Postman
   - Update API client in `frontend/api-client.ts`

2. **Frontend Integration:**
   - Add UI components in `frontend/index.tsx`
   - Connect to backend via `backend-service.ts`
   - Test in browser

3. **Documentation:**
   - Update this file with new features
   - Add inline code comments
   - Update README if user-facing

### For Connecting Existing Scripts:

1. **Wrap in API Endpoint:**
   - Import the script/module in `api/main.py`
   - Create endpoint with proper request/response models
   - Handle async execution if needed

2. **Add UI Control:**
   - Create panel/button in frontend
   - Add parameter inputs if needed
   - Show results/progress

## Installation & Setup Notes

### Environment Setup
```bash
# Python environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r api/requirements.txt

# Node.js environment
npm install

# Environment variables
cp frontend/.env.example frontend/.env.local
cp api/.env.example api/.env
```

### Running Development Servers
```bash
# Both servers at once
npm run dev

# Or separately:
npm run dev:frontend  # Frontend on :3000
npm run dev:backend   # Backend on :8000
```

### Building for Production
```bash
npm run build
# Then serve frontend/dist with your web server
# Run backend with gunicorn or similar WSGI server
```

## Resources & References

- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **React Docs:** https://react.dev/
- **Vite Docs:** https://vitejs.dev/
- **FracFire Generator:** See `src/core/generator/` for implementation details
- **Setup Detection:** See `src/core/detector/` for algorithms

## Contributing Guidelines

When adding new features or refactoring:

1. Keep the backend API RESTful and well-documented
2. Maintain type safety in both Python and TypeScript
3. Add error handling at every layer
4. Write tests for critical functionality
5. Update this document with your changes
6. Keep frontend components modular and reusable

## Questions & Support

For questions about the integration or to report issues, please:
- Check existing issues in the GitHub repository
- Review the code comments in `api/main.py` and `frontend/index.tsx`
- Refer to the original fracfire documentation in `docs/`

---

**Last Updated:** 2025-12-07  
**Integration Status:** Phase 2-3 Complete, Phase 4-6 Pending
