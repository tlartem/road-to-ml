"""Feast feature definitions for NYC Taxi pipeline."""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64

# Entity: taxi zone (used for both pickup and dropoff lookups)
zone = Entity(
    name="zone",
    join_keys=["zone_id"],
)

# Source: pre-computed zone-level statistics (Parquet on PVC)
zone_stats_source = FileSource(
    path="/feast-data/features/zone_stats.parquet",
    timestamp_field="event_timestamp",
)

# Feature view: aggregated stats per zone
zone_stats = FeatureView(
    name="zone_stats",
    entities=[zone],
    ttl=timedelta(days=90),
    schema=[
        Field(name="zone_avg_fare", dtype=Float64),
        Field(name="zone_avg_duration_min", dtype=Float64),
        Field(name="zone_avg_distance", dtype=Float64),
        Field(name="zone_trip_count", dtype=Int64),
    ],
    source=zone_stats_source,
)
