const fs = require('fs');
const path = require('path');

// Simple file-backed database storage that requires ZERO native C++ builds
const DB_FILE = path.join(__dirname, 'sentiment_records.json');

class DatabaseService {
  constructor() {
    this.records = [];
    this.load();
  }

  load() {
    try {
      if (fs.existsSync(DB_FILE)) {
        const data = fs.readFileSync(DB_FILE, 'utf8');
        this.records = JSON.parse(data) || [];
      }
    } catch (e) {
      this.records = [];
    }
  }

  save() {
    try {
      fs.writeFileSync(DB_FILE, JSON.stringify(this.records.slice(-500), null, 2), 'utf8');
    } catch (e) {
      console.warn('Failed to save records:', e.message);
    }
  }

  addRecord(record) {
    const entry = {
      id: this.records.length + 1,
      created_at: new Date().toISOString(),
      ...record
    };
    this.records.push(entry);
    this.save();
    return entry;
  }
}

module.exports = new DatabaseService();
