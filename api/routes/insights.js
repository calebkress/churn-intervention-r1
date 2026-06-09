/**
 * api/routes/insights.js
 * ======================
 * Proxies /api/ml/* → FastAPI at port 8000.
 *
 * Routes handled here:
 *   GET /api/ml/insights/:id   → FastAPI GET /insights/:id
 *   GET /api/ml/similar/:id    → FastAPI GET /similar/:id
 *   GET /api/ml/infer/:id      → FastAPI POST /infer  (builds obs from customer first)
 *   GET /api/ml/health         → FastAPI GET /health
 *
 */

const express = require('express');
const mongoose = require('mongoose');
const axios = require('axios');
const router = express.Router();

const FASTAPI_URL = process.env.FASTAPI_URL || 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Customer model (reuse — Mongoose caches models by name)
// ---------------------------------------------------------------------------

const CustomerSchema = new mongoose.Schema({
  customer_id:              String,
  tenure_months:            Number,
  plan_type:                String,
  monthly_spend:            Number,
  avg_monthly_data_gb:      Number,
  call_drop_rate:           Number,
  support_tickets_90d:      Number,
  payment_failures_90d:     Number,
  nps_score:                Number,
  days_since_last_contact:  Number,
  churn_probability:        Number,
  churn_label:              Boolean,
  feature_vector:           [Number],
}, { collection: 'customers', versionKey: false });

const Customer = mongoose.models.Customer
  || mongoose.model('Customer', CustomerSchema);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PLAN_TYPE_ENCODING = { prepaid: 0, postpaid: 1, enterprise: 2 };

/**
 * Build the 14-dim observation vector from a customer document.
 * Field order must match churn_env.py to_observation() exactly.
 */
function buildObservation(customer) {
  return [
    Math.min(customer.tenure_months / 120.0, 1.0),
    (PLAN_TYPE_ENCODING[customer.plan_type] ?? 1) / 2.0,
    Math.min(customer.monthly_spend / 300.0, 1.0),
    Math.min(customer.avg_monthly_data_gb / 50.0, 1.0),
    Math.min(customer.call_drop_rate, 1.0),
    Math.min(customer.support_tickets_90d / 10.0, 1.0),
    Math.min(customer.payment_failures_90d / 5.0, 1.0),
    customer.nps_score / 10.0,
    Math.min(customer.days_since_last_contact / 180.0, 1.0),
    customer.churn_probability,
    0.0,   // interventions_this_episode — 0 at inference time
    0.0,   // last_action — 0 at inference time
    1.0,   // steps_remaining_normalized — full episode at inference time
    customer.churn_probability >= 0.6 ? 1.0    // risk_level
      : customer.churn_probability >= 0.3 ? 0.5
      : 0.0,
  ];
}

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/**
 * GET /api/ml/health
 * Check that FastAPI is up and the model is loaded.
 */
router.get('/health', async (req, res, next) => {
  try {
    const { data } = await axios.get(`${FASTAPI_URL}/health`);
    res.json(data);
  } catch (err) {
    res.status(503).json({ error: 'FastAPI unavailable', detail: err.message });
  }
});

/**
 * GET /api/ml/insights/:id
 * LangChain recommendation for a customer.
 * Proxies to FastAPI GET /insights/{customer_id}.
 */
router.get('/insights/:id', async (req, res, next) => {
  try {
    const { data } = await axios.get(`${FASTAPI_URL}/insights/${req.params.id}`);
    res.json(data);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});

/**
 * GET /api/ml/similar/:id
 * Raw Vector Search results for a customer (no LLM).
 * Proxies to FastAPI GET /similar/{customer_id}.
 */
router.get('/similar/:id', async (req, res, next) => {
  try {
    const n = req.query.n || 5;
    const { data } = await axios.get(`${FASTAPI_URL}/similar/${req.params.id}?n=${n}`);
    res.json(data);
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});

/**
 * GET /api/ml/infer/:id
 * Run the PPO policy on a customer.
 *
 * Fetches the customer from Atlas, builds the 14-dim observation,
 * then POSTs to FastAPI /infer. Returns the recommended action.
 */
router.get('/infer/:id', async (req, res, next) => {
  try {
    // Fetch customer from Atlas
    const customer = await Customer
      .findOne({ customer_id: req.params.id }, { _id: 0 })
      .lean();

    if (!customer) {
      return res.status(404).json({ error: `Customer ${req.params.id} not found` });
    }

    // Build observation vector
    const observation = buildObservation(customer);

    // POST to FastAPI
    const { data } = await axios.post(`${FASTAPI_URL}/infer`, { observation });

    res.json({
      customer_id: req.params.id,
      ...data,
    });
  } catch (err) {
    if (err.response) {
      return res.status(err.response.status).json(err.response.data);
    }
    next(err);
  }
});

module.exports = router;