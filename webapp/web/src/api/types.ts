export interface User {
  email: string;
  role: "admin" | "operator" | "viewer";
  tenant_id: string;
}

export interface PersonSummary {
  person_id: string;
  status: string | null;
  first_seen_cam: number | null;
  last_seen_cam: number | null;
  first_seen_time: string | null;
  last_seen_time: string | null;
  gallery_size: number | null;
  has_description: boolean;
  name: string | null;
  gender: string | null;
  age_range: string | null;
  ethnicity: string | null;
  glasses: string | null;
  snapshot_count: number;
  thumbnail_url: string | null;
  score?: number | null;
  summary?: string | null;
}

export interface GalleryEntry {
  id: number;
  embedding_type: string;
  angle_tag: string;
  source_cam: number | null;
  captured_at: string;
}

export interface GalleryMeta {
  count: number;
  dim: number | null;
  by_type: Record<string, number>;
  by_camera: Record<string, number>;
  entries: GalleryEntry[];
}

export interface JourneyStop {
  id: number;
  cam_id: number;
  track_id: number;
  first_seen: string | null;
  last_seen: string | null;
}

export interface Journey {
  person_id: string;
  cameras: number[];
  overlap_groups: number[][];
  stops: JourneyStop[];
}

export interface PersonDescription {
  summary: string | null;
  attributes: Record<string, unknown>;
  described_at?: string;
  model_id?: string;
}

export interface PersonDetail {
  person_id: string;
  status: string | null;
  first_seen_cam: number | null;
  first_seen_time: string | null;
  last_seen_cam: number | null;
  last_seen_time: string | null;
  gallery_size: number | null;
  known_angles: string[];
  latest_description_id: number | null;
  created_at: string | null;
  name: string | null;
  gender: string | null;
  age_range: string | null;
  ethnicity: string | null;
  glasses: string | null;
  cameras: number[];
  gallery: GalleryMeta;
  journey: Journey;
  description: PersonDescription | null;
  snapshots: string[];
}

export interface Stats {
  persons: number;
  by_status: Record<string, number>;
  described: number;
  undescribed: number;
  multi_camera: number;
  total_body_embeddings: number;
  per_camera_sightings: Record<string, number>;
  distributions: {
    gender: Record<string, number>;
    ethnicity: Record<string, number>;
    glasses: Record<string, number>;
  };
  alerts: number;
}

export interface Camera {
  cam_id: number;
  name: string;
  available: boolean;
  stream_url: string;
  overlap_group: number[] | null;
  overlay_available?: boolean;
}

export interface AlertItem {
  timestamp?: string;
  cam_id?: number;
  score?: number;
  level?: string;
  label?: string;
}

export interface AuditItem {
  id: number;
  ts: string;
  user_email: string;
  action: string;
  target_ids: string[] | null;
  detail: Record<string, unknown> | null;
}

export interface SearchResponse {
  count: number;
  results: PersonSummary[];
  note?: string | null;
  query?: string;
  mode?: string;
}
