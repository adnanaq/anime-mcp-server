# 🔄 Anime Database Update Strategy

## Overview

The anime-offline-database updates weekly, so we need an efficient strategy to keep our vector database synchronized without full re-indexing.

## 📊 Update Types

### 1. **Incremental Updates** (Recommended)
- **Frequency**: Weekly (automated)
- **Duration**: 10-30 minutes 
- **Process**: Only index new/changed entries
- **Efficiency**: 95%+ time savings vs full update

### 2. **Full Updates** (Emergency only)
- **Frequency**: Manual or quarterly
- **Duration**: 2-3 hours
- **Process**: Complete re-index of all 38,894 entries
- **Use Cases**: Major schema changes, corruption recovery

## 🤖 Automated Workflow

```
Weekly Schedule (Every Friday 2 AM):
1. Check for updates → Compare content hash
2. If changes detected → Download new data  
3. Compare datasets → Find added/modified/removed entries
4. Process changes → Generate embeddings for new entries
5. Update vector DB → Add new, update modified, remove deleted
6. Log results → Track changes and performance

Note: anime-offline-database typically updates Tuesday-Thursday,
so Friday 2 AM ensures we catch the latest weekly changes.
```

## 📡 API Endpoints

### Admin Endpoints
```http
POST /api/admin/check-updates          # Check for available updates (✅ Implemented)
POST /api/admin/download-data          # Download latest anime database
POST /api/admin/process-data           # Process and index data into Qdrant
GET  /api/admin/update-status         # Get update status & metadata
POST /api/admin/schedule-weekly-update # Manually trigger weekly job
```

### Usage Examples
```bash
# Check if updates are available
curl -X POST http://localhost:8000/api/admin/check-updates

# Download latest data
curl -X POST http://localhost:8000/api/admin/download-data

# Process and index data
curl -X POST http://localhost:8000/api/admin/process-data

# View database stats
curl http://localhost:8000/stats
```

## 🔧 Setup Instructions

### 1. Automatic Updates (Recommended)
```bash
# Setup weekly cron job
./scripts/setup_cron.sh

# Verify cron job
crontab -l
```

### 2. Manual Updates
```bash
# Check for updates
python scripts/weekly_update.py

# Or via API
curl -X POST http://localhost:8000/api/admin/schedule-weekly-update
```

## 📊 Performance Comparison

| Update Type | Duration | Data Processed | Downtime | Efficiency |
|-------------|----------|----------------|----------|------------|
| **Incremental** | 10-30 min | ~100-500 entries | None | ⭐⭐⭐⭐⭐ |
| **Full** | 2-3 hours | 38,894 entries | Partial | ⭐⭐ |

## 🛡️ Smart Diffing Algorithm

The incremental update uses intelligent comparison:

1. **Content Hashing**: MD5 hash of entire dataset
2. **Entry-Level Comparison**: Hash of key fields per anime
3. **Change Detection**: Added, modified, removed entries
4. **Vector Efficiency**: Only re-embed changed content

### Example Weekly Changes
```json
{
  "added": 15,        // New anime releases
  "modified": 8,      // Updated episode counts, status
  "removed": 2,       // Duplicate cleanup
  "total_time": "12 minutes"
}
```

## 📁 File Structure

```
data/
├── raw/
│   ├── anime-offline-database.json         # Current data
│   └── anime-offline-database-latest.json  # New download
├── processed/
│   └── anime-vectors.json                  # Vector-ready data
└── update_metadata.json                    # Update tracking
```

## 🔍 Monitoring

### Log Files
```bash
# Weekly update logs
tail -f logs/weekly_update.log

# Cron job logs  
tail -f logs/cron.log

# Server logs
tail -f server.log
```

### Status Tracking
```bash
# Current database stats
curl http://localhost:8000/stats

# Update metadata
curl http://localhost:8000/api/admin/update-status
```

## 🚨 Troubleshooting

### Common Issues

1. **Update Fails**
   - Check Qdrant connection
   - Verify disk space
   - Review error logs

2. **Slow Performance**
   - Reduce batch size
   - Check memory usage
   - Monitor network latency

3. **Data Inconsistency**
   - Run full update
   - Clear index and re-index
   - Verify source data integrity

### Emergency Procedures

```bash
# Force full update
curl -X POST http://localhost:8000/api/admin/update-full

# Clear and rebuild index
# (Implement index clearing in future version)

# Rollback to previous data
# (Backup strategy needed for production)
```

## 🔮 Future Enhancements

### Phase 2 Features
- [ ] **Backup & Rollback**: Snapshot vector index before updates
- [ ] **Health Monitoring**: Automated alert system
- [ ] **Delta Compression**: Store only changes for efficiency
- [ ] **Multi-Source Sync**: Support additional anime databases
- [ ] **Real-time Updates**: WebSocket notifications for changes

### Phase 3 Features
- [ ] **Blue-Green Deployment**: Zero-downtime updates
- [ ] **Distributed Updates**: Multi-node update coordination
- [ ] **ML-Powered Diffing**: Intelligent change detection
- [ ] **User Impact Analysis**: Track how updates affect search quality

## 📈 Success Metrics

- **Update Frequency**: Weekly (52 updates/year)
- **Downtime**: <1 minute during incremental updates
- **Data Freshness**: <7 days lag from source
- **Error Rate**: <1% failed updates
- **Performance**: <200ms search response times maintained

---

**Current Status**: ✅ Basic update system implemented with Qdrant integration. Incremental updates and automation scheduled for Phase 4 enhancements.