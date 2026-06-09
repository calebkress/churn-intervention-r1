/**
 * api/routes/interventions.js
 * ===========================
 * GET /api/interventions                    — all interventions
 * GET /api/interventions/by-customer/:id    — interventions for one customer
 *
 */

const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const InterventionSchema = new mongoose.Schema({
  intervention_id:    { type: String, required: true },
  customer_id:        { type: String, required: true, index: true },
  type:               String,   // action name e.g. "outbound_call"
  date:               String,
  outcome:            String,   // "retained" or "churned"
  agent_action_index: Number,   // 0-4
}, { collection: 'interventions', versionKey: false });

const Intervention = mongoose.models.Intervention
  || mongoose.model('Intervention', InterventionSchema);

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/**
 * GET /api/interventions
 * Returns all intervention records.
 *
 * Query params:
 *   outcome  — filter by "retained" or "churned" (optional)
 *   type     — filter by intervention type e.g. "outbound_call" (optional)
 *   limit    — max results (default 1000, max 5000)
 */
router.get('/', async (req, res, next) => {
  try {
    const limit = Math.min(5000, parseInt(req.query.limit) || 1000);

    const filter = {};
    if (req.query.outcome) filter.outcome = req.query.outcome;
    if (req.query.type)    filter.type    = req.query.type;

    const interventions = await Intervention
      .find(filter, { _id: 0 })
      .limit(limit)
      .lean();

    res.json({ interventions, total: interventions.length });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/interventions/by-customer/:id
 * Returns all interventions for a specific customer, sorted by date.
 */
router.get('/by-customer/:id', async (req, res, next) => {
  try {
    const interventions = await Intervention
      .find({ customer_id: req.params.id }, { _id: 0 })
      .sort({ date: 1 })
      .lean();

    res.json({ customer_id: req.params.id, interventions });
  } catch (err) {
    next(err);
  }
});

module.exports = router;