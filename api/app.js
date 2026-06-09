/**
 * api/app.js
 * ==========
 * Express entry point. Runs on port 3001.
 *
 * RUN:
 *   npm run dev       (nodemon — auto-restarts on file changes)
 *   npm start         (plain node — for production)
 *
 * REQUIRES:
 *   FastAPI running on port 8000 for /api/ml/* routes to work.
 *   .env with MONGODB_URI, MONGODB_DB, FASTAPI_URL, PORT.
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { connectAtlas } = require('./db/atlas');

const customersRouter = require('./routes/customers');
const interventionsRouter = require('./routes/interventions');
const trainingRunsRouter = require('./routes/training_runs');
const insightsRouter = require('./routes/insights');

const app = express();
const PORT = process.env.PORT || 3001;

// --- Middleware ---
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:3000'],  // Vite dev server
}));
app.use(express.json());

// --- Routes ---
app.use('/api/customers', customersRouter);
app.use('/api/interventions', interventionsRouter);
app.use('/api/training-runs', trainingRunsRouter);
app.use('/api/ml', insightsRouter);  // proxies to FastAPI

// --- Health check ---
app.get('/health', (req, res) => {
  res.json({ status: 'ok', port: PORT });
});

// --- 404 handler ---
app.use((req, res) => {
  res.status(404).json({ error: `Route not found: ${req.method} ${req.path}` });
});

// --- Error handler ---
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: err.message || 'Internal server error' });
});

// --- Start ---
async function start() {
  console.log('\nStarting Express...');
  try {
    await connectAtlas();
  } catch (err) {
    console.error('  Atlas connection failed:', err.message);
    process.exit(1);
  }
  app.listen(PORT, () => {
    console.log(`  Express running on http://localhost:${PORT}`);
    console.log(`  FastAPI expected at ${process.env.FASTAPI_URL || 'http://localhost:8000'}\n`);
  });
}

start();