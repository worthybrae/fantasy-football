# Fantasy Football Draft Tool

A comprehensive fantasy football draft tool that combines data refresh, scoring calculations, and an interactive web-based draft interface.

## Workflow

1. **refresh** - Download and update player statistics from NFL Data
2. **api** - Compute PPR points, scoring factors, value over replacement, and build draft tier recommendations
3. **web** - Interactive draft board UI for live draft management

## Project Structure

- `pipeline/` - Data ingestion and refresh workflows
- `scoring/` - PPR calculations, scoring factors, and player valuations
- `api/` - FastAPI backend serving draft data and recommendations
- `tests/` - Test suite

(To be expanded in Task 9)