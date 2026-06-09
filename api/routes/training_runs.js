/**
 * api/routes/training_runs.js
 * ===========================
 * GET /api/training-runs      — all training runs, newest first
 * GET /api/training-runs/:id  — single training run by run_id
 *
 */

const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const TrainingRunSchema = new mongoose.Schema({
  run_id:                 { type: String, required: true },
  mlflow_run_id:          String,
  timestamp:              String,
  algorithm:              String,
  hyperparameters:        mongoose.Schema.Types.Mixed,
  n_customers:            Number,
  churn_rate_baseline:    Number,
  churn_rate_trained:     Number,
  churn_rate_reduction:   Number,
  mean_eval_reward:       Number,
  reward_curve:           [mongoose.Schema.Types.Mixed],
  intervention_distribution: mongoose.Schema.Types.Mixed,
  model_path:             String,
}, { collection: 'training_runs', versionKey: false });

const TrainingRun = mongoose.models.TrainingRun
  || mongoose.model('TrainingRun', TrainingRunSchema);

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/**
 * GET /api/training-runs
 * Returns all training runs sorted newest first.
 * Excludes reward_curve by default (large array) — fetch single run for that.
 *
 * Query params:
 *   full — set to "true" to include reward_curve arrays
 */
router.get('/', async (req, res, next) => {
  try {
    const projection = req.query.full === 'true'
      ? { _id: 0 }
      : { _id: 0, reward_curve: 0 };  // omit large array from list view

    const runs = await TrainingRun
      .find({}, projection)
      .sort({ timestamp: -1 })
      .lean();

    res.json({ runs, total: runs.length });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/training-runs/:id
 * Returns a single training run including the full reward_curve array.
 * The React RewardCurve component calls this to get chart data.
 */
router.get('/:id', async (req, res, next) => {
  try {
    const run = await TrainingRun
      .findOne({ run_id: req.params.id }, { _id: 0 })
      .lean();

    if (!run) {
      return res.status(404).json({ error: `Training run ${req.params.id} not found` });
    }

    res.json(run);
  } catch (err) {
    next(err);
  }
});

module.exports = router;