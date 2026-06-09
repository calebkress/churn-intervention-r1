/**
 * api/routes/customers.js
 * =======================
 * GET /api/customers        — paginated customer list
 * GET /api/customers/:id    — single customer by customer_id
 *
 */

const express = require('express');
const mongoose = require('mongoose');
const router = express.Router();

// ---------------------------------------------------------------------------
// Schema — mirrors data/atlas.py customer document shape
// ---------------------------------------------------------------------------

const CustomerSchema = new mongoose.Schema({
  customer_id:              { type: String, required: true, unique: true },
  tenure_months:            Number,
  plan_type:                String,
  monthly_spend:            Number,
  contract_end_date:        String,
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

// Avoid re-registering model on hot reload
const Customer = mongoose.models.Customer || mongoose.model('Customer', CustomerSchema);

// ---------------------------------------------------------------------------
// Routes
// ---------------------------------------------------------------------------

/**
 * GET /api/customers
 * Returns a paginated list of customers, sorted by churn_probability desc.
 *
 * Query params:
 *   page  — page number, 1-indexed (default: 1)
 *   limit — results per page (default: 50, max: 200)
 *   plan  — filter by plan_type (optional)
 *   churn — filter by churn_label true/false (optional)
 */
router.get('/', async (req, res, next) => {
  try {
    const page  = Math.max(1, parseInt(req.query.page)  || 1);
    const limit = Math.min(200, parseInt(req.query.limit) || 50);
    const skip  = (page - 1) * limit;

    const filter = {};
    if (req.query.plan)  filter.plan_type    = req.query.plan;
    if (req.query.churn !== undefined) {
      filter.churn_label = req.query.churn === 'true';
    }

    const [customers, total] = await Promise.all([
      Customer.find(filter, { _id: 0, feature_vector: 0 })  // exclude large array from list view
        .sort({ churn_probability: -1 })
        .skip(skip)
        .limit(limit)
        .lean(),
      Customer.countDocuments(filter),
    ]);

    res.json({
      customers,
      pagination: { page, limit, total, pages: Math.ceil(total / limit) },
    });
  } catch (err) {
    next(err);
  }
});

/**
 * GET /api/customers/:id
 * Returns a single customer including feature_vector (needed for /ml/insights).
 */
router.get('/:id', async (req, res, next) => {
  try {
    const customer = await Customer.findOne(
      { customer_id: req.params.id },
      { _id: 0 }
    ).lean();

    if (!customer) {
      return res.status(404).json({ error: `Customer ${req.params.id} not found` });
    }

    res.json(customer);
  } catch (err) {
    next(err);
  }
});

module.exports = router;