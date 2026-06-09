/**
 * api/db/atlas.js
 * ===============
 * Opens and exports the Mongoose connection to MongoDB Atlas.
 *
 * USAGE:
 *   Called once in api/app.js at startup. Routes don't import this directly —
 *   they use the Mongoose models defined alongside each route.
 */

const mongoose = require('mongoose');

async function connectAtlas() {
  const uri = process.env.MONGODB_URI;
  if (!uri) {
    throw new Error('MONGODB_URI is not set in your .env file.');
  }

  await mongoose.connect(uri, {
    dbName: process.env.MONGODB_DB || 'churn_rl',
  });

  console.log(`  Atlas connected: ${mongoose.connection.db.databaseName}`);
}

module.exports = { connectAtlas };