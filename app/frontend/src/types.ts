export type Page = "new" | "maps" | "vehicles" | "plugins" | "settings" | "thread";

export interface ModelEntry {
  id: string;
  model: string;
  label: string;
  provider: string;
  icon: string;
  source: "default" | "custom";
  profile_id?: string;
}

export interface CustomModelDiscovery {
  provider: string;
  icon: string;
  base_url: string;
  models: string[];
  warning?: string;
}

export interface Message {
  message_id: string;
  sequence: number;
  role: "user" | "assistant" | "system";
  kind: "text" | "status" | "plan" | "error";
  content: string;
  metadata: Record<string, unknown>;
  created_at: string;
}

export interface TaskThread {
  thread_id: string;
  title: string;
  state: string;
  selected_model: string;
  selected_map_id: string | null;
  selected_map_content_sha256: string | null;
  selected_vehicle_id: string | null;
  selected_vehicle_content_sha256: string | null;
  locale: "zh-CN" | "en-US";
  pinned: boolean;
  archived: boolean;
  created_at: string;
  updated_at: string;
  messages?: Message[];
}

export type AssetImportState = "created" | "quarantining" | "parsing" | "needs_input" | "normalizing" | "building" | "validating" | "qualified" | "failed" | "cancelled";

export interface AssetImportJob {
  schema_version: "dronedream.asset-import-job.v1";
  job_id: string;
  owner_id: string;
  source_name: string;
  source_format: string;
  detected_source_format: string | null;
  source_adapter_id: string | null;
  package_sha256: string;
  normalized_content_sha256: string | null;
  qualified_content_sha256: string | null;
  state: AssetImportState;
  progress_percent: number;
  revision: number;
  asset_id: string | null;
  asset_kind: "map" | "world" | "vehicle" | null;
  issue_codes: string[];
  required_inputs: string[];
  created_at: string;
  updated_at: string;
}

export interface AssetIssue {
  schema_version: "dronedream.asset-issue.v1";
  code: string;
  severity: "warning" | "error" | "critical";
  stage: string;
  location: string;
  title: { "zh-CN": string; "en-US": string };
  detail: { "zh-CN": string; "en-US": string };
  actions: Array<{ id: string; "zh-CN": string; "en-US": string }>;
}

export interface AssetIssueReport {
  schema_version: "dronedream.asset-issue-report.v1";
  job_id: string;
  state: string;
  progress_percent: number;
  issues: AssetIssue[];
}

export interface AssetVersionEntry {
  asset_id: string;
  content_sha256: string;
  kind: "map" | "world" | "vehicle";
  maturity: "visual_only" | "physics_ready" | "simulation_ready" | "flight_ready" | "qualified";
  bundle_root: string;
  manifest: Record<string, unknown>;
  asset_ir: { name?: string; version?: string } & Record<string, unknown>;
  imported_at: string;
}

export type AssetQualificationState = "created" | "preparing" | "running" | "validating" | "paused" | "qualified" | "failed" | "cancelled";

export interface AssetQualificationJob {
  schema_version: "dronedream.asset-pair-qualification-job.v1";
  job_id: string;
  map_asset_id: string;
  map_content_sha256: string;
  vehicle_asset_id: string;
  vehicle_content_sha256: string;
  state: AssetQualificationState;
  progress_percent: number;
  qualification_id: string | null;
  result_map_content_sha256: string | null;
  result_vehicle_content_sha256: string | null;
  issue_codes: string[];
  cancel_requested: boolean;
  created_at: string;
  updated_at: string;
}

export interface PortablePluginSnapshot {
  schema_version: "dronedream.plugin-snapshot.v1";
  snapshot_id: string;
  catalog_sha256: string;
  created_at: string;
  plugins: Array<{
    plugin_id: string;
    version: string;
    package_sha256: string;
    manifest_sha256: string;
    configuration_sha256: string;
    configuration: Record<string, unknown>;
    capability_ids: string[];
    manifest?: Record<string, unknown> | null;
    bundle_root: null;
  }>;
}

export interface AssetQualificationEvidence {
  schema_version: "dronedream.asset-qualification-evidence.v1";
  job_id: string;
  qualification_id: string;
  map_asset_id: string;
  map_content_sha256: string;
  vehicle_asset_id: string;
  vehicle_content_sha256: string;
  evidence_sha256: string;
  runtime_contracts: {
    schema_version: "dronedream.asset-pair-runtime-contracts.v1";
    map: {
      asset_id: string;
      content_sha256: string;
      coordinate_frame: "ENU";
      node_count: number;
      edge_count: number;
      named_entity_count: number;
      navigation_bounds_m: {
        minimum: { x: number; y: number; z: number };
        maximum: { x: number; y: number; z: number };
        span: { x: number; y: number; z: number };
      };
      semantic_layers: string[];
      simulation_targets: AssetRuntimeSimulationTarget[];
    };
    vehicle: {
      schema_version: "dronedream.vehicle.v1";
      asset_id: string;
      content_sha256: string;
      name: string;
      coordinate_frame: "base_link_frd";
      dry_mass_kg: number;
      max_takeoff_mass_kg: number;
      body_radius_m: number;
      body_height_m: number;
      max_speed_mps: number;
      max_acceleration_mps2: number;
      qualified_range_m: number;
      reserve_battery_percent: number;
      max_pickup_payload_kg: number;
      sensors: string[];
      vehicle_class: "multirotor" | "fixed_wing" | "vtol" | "ground" | "other" | "unknown";
      simulation_targets: AssetRuntimeSimulationTarget[];
    };
  };
  receipt: {
    plugin_snapshot?: PortablePluginSnapshot | null;
    plugin_snapshot_sha256?: string | null;
    plugin_checks?: Array<{
      check_id: string;
      plugin_id: string;
      capability_id: string;
      accepted: boolean;
      issue_codes: string[];
      details: Record<string, unknown>;
    }>;
    plugin_hook_receipts?: Array<Record<string, unknown>>;
    runtime_evidence?: {
      gates?: Record<string, boolean>;
      measurements?: Record<string, unknown>;
    };
    [key: string]: unknown;
  };
}

export interface AssetRuntimeSimulationTarget {
  target_id: string;
  simulator: "gazebo-classic" | "gazebo-harmonic" | "isaac-sim" | "webots" | "other";
  simulator_version: string;
  ros_distribution: string | null;
  autopilot: "px4" | "ardupilot" | "none" | "other";
  entrypoint: string;
}

export interface AssetSourceAdapter {
  adapter_id: string;
  name: string;
  version: string;
  availability: "builtin" | "companion_required" | "plugin_required";
  source_formats: string[];
  file_extensions: string[];
  asset_kinds: Array<"map" | "world" | "vehicle">;
  output_format: "ddpkg";
  execution_boundary: "declarative-parser" | "isolated-local-companion" | "isolated-plugin";
  required_application: string | null;
  documentation_url: string | null;
  enabled: boolean;
  provider_plugin_ids: string[];
}

export interface PluginEntry {
  plugin_id: string;
  name: string;
  version: string;
  authority: string;
  enabled: boolean;
  builtin: boolean;
  description: string;
  publisher: string;
  runtime_kind: string;
  status: string;
  health: string;
  removable: boolean;
  disable_allowed: boolean;
  slot_required: boolean;
  package_sha256: string;
  last_error: string | null;
  trust_status: "verified" | "local-approved" | "unverified" | "revoked";
  trust_decision: {
    status?: string;
    issue_codes?: string[];
    publisher_key_id?: string | null;
    manifest_sha256?: string;
    package_sha256?: string;
  };
  update_ring: "stable" | "preview" | "canary" | "pinned";
  capabilities: Array<{
    capability_id: string;
    kind: string;
    name: string;
    description: string;
    authority: string;
    metadata: Record<string, unknown>;
  }>;
  permissions: string[];
  dependencies: Array<{ plugin_id: string; version: string; optional: boolean }>;
  placement: {
    category_id: string;
    category_label: string;
    slot_id: string;
    slot_label: string;
    activation_mode: "single" | "multiple" | "pipeline";
    scope: "general" | "mission" | "runtime" | "interface";
    failure_mode: "fail-closed" | "isolate" | "advisory";
    swap_policy: "anytime" | "next-mission" | "safe-hold" | "restart" | "certified-update";
    category_order: number;
    slot_order: number;
    plugin_order: number;
    pipeline_order: number;
    runs_after: string[];
    runs_before: string[];
  };
  manifest: Record<string, unknown>;
}

export interface PluginDetail extends PluginEntry {
  versions: Array<{
    plugin_id: string;
    version: string;
    package_sha256: string;
    installed_at: string;
    trust_status: "verified" | "local-approved" | "unverified" | "revoked";
  }>;
  events: Array<{
    receipt_id: string;
    operation: string;
    accepted: boolean;
    created_at: string;
  }>;
  configuration: Record<string, unknown>;
  governance_decisions: Array<{
    decision_id: string;
    operation: string;
    accepted: boolean;
    issue_codes: string[];
    created_at: string;
  }>;
  usage: PluginUsageEvent[];
  usage_summary: {
    calls: number;
    successes: number;
    errors: number;
    duration_ms: number;
    average_duration_ms: number;
    input_bytes: number;
    output_bytes: number;
    last_called_at: string | null;
  };
}

export interface PluginUsageEvent {
  invocation_id: string;
  plugin_id: string;
  plugin_version: string;
  capability_id: string;
  slot_id: string;
  invocation_kind: "tool" | "hook";
  outcome: "success" | "error";
  duration_ms: number;
  input_bytes: number;
  output_bytes: number;
  issue_code: string | null;
  created_at: string;
}

export interface PluginGovernancePolicy {
  schema_version: "dronedream.plugin-governance-policy.v1";
  policy_id: string;
  mode: "personal" | "managed";
  allowed_plugin_ids: string[];
  allowed_publishers: string[];
  denied_permissions: string[];
  allowed_update_rings: Array<"stable" | "preview" | "canary" | "pinned">;
  require_verified_signatures: boolean;
  allow_local_approval: boolean;
  maximum_external_plugins: number;
}

export interface PluginGovernanceOverview {
  policy: PluginGovernancePolicy;
  decisions: PluginDetail["governance_decisions"];
}

export interface PluginMarketplaceSource {
  schema_version: "dronedream.plugin-marketplace-source.v1";
  source_id: string;
  name: string;
  index_url: string;
  enabled: boolean;
}

export interface PluginMarketplaceCatalog {
  sources: PluginMarketplaceSource[];
  entries: Array<{
    source_id: string;
    source_name: string;
    plugin_id: string;
    version: string;
    name: string;
    description: string;
    publisher: string;
    archive_url: string;
    archive_sha256: string;
    category_id: string;
    update_ring: string;
  }>;
  errors: Array<{ source_id: string; issue_code: string }>;
}

export interface PluginPanel {
  schema_version: "dronedream.ui-panel.v1";
  title: string;
  sections: Array<{
    section_id: string;
    title: string;
    widgets: Array<{
      widget_id: string;
      kind: "text" | "status" | "metric" | "log" | "table" | "replay" | "telemetry" | "configuration-form";
      label: string;
      source: string;
      path: string;
      value?: string | number | boolean | null;
      resolved?: unknown;
      unit?: string | null;
      schema?: Record<string, unknown>;
    }>;
    actions: Array<{
      action_id: "panel.refresh" | "plugin.healthcheck" | "plugin.disable";
      label: string;
      style: "default" | "primary" | "danger";
      confirmation?: string | null;
    }>;
  }>;
}

export interface ConnectorCredentialReference {
  reference: string;
  display_name: string;
  allowed_plugin_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface Bootstrap {
  models: ModelEntry[];
  threads: TaskThread[];
  asset_import_jobs: AssetImportJob[];
  asset_versions: AssetVersionEntry[];
  asset_qualification_jobs: AssetQualificationJob[];
  asset_source_adapters: AssetSourceAdapter[];
  plugins: PluginEntry[];
  connector_credentials: ConnectorCredentialReference[];
  settings: {
    locale: string;
    theme: string;
    update_channel: string;
    default_model_id: string;
    memory_enabled: boolean;
    remember_task_preferences: boolean;
    remember_asset_choices: boolean;
    last_map_id: string | null;
    last_map_content_sha256: string | null;
    last_vehicle_id: string | null;
    last_vehicle_content_sha256: string | null;
    plugin_update_ring: string;
    plugin_governance: PluginGovernancePolicy;
    plugin_marketplace_sources: PluginMarketplaceSource[];
  };
}

export type ManagedPlanId = "free" | "plus" | "pro";

export interface ManagedUsageSnapshot {
  plan: {
    id: ManagedPlanId;
    name: string;
    monthly_price_cny_fen: number;
    included_ai_credits: number;
    capability_set: string;
  };
  account?: {
    billing_scope: "individual" | "business";
    organization_id: string | null;
    organization_name: string | null;
    organization_role: "owner" | "admin" | "member" | null;
  };
  period: { starts_at: string; ends_at: string };
  usage: {
    reserved_ai_credits: number;
    consumed_ai_credits: number;
    remaining_ai_credits: number;
    request_count: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    estimated_request_count: number;
    credit_policy_version: number;
  };
  daily_usage?: Array<{
    date: string;
    consumed_ai_credits: number;
    request_count: number;
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
  }>;
}

export interface DailyModelUsage {
  date: string;
  aiCredits: number;
  totalTokens: number;
  requestCount: number;
}

export interface AccountOverview {
  displayName: string;
  email: string;
  avatarUrl: string | null;
  snapshot: ManagedUsageSnapshot;
  dailyUsage: DailyModelUsage[];
  dailyUsageComplete: boolean;
}

export interface RuntimeStatus {
  distribution: string;
  runtime_available: boolean;
  resources_ready: boolean;
  provisioned: boolean;
  issue: string | null;
}

export type RuntimeSetupPhase =
    | "idle"
    | "queued"
    | "validatingResources"
    | "checkingBaseRuntime"
    | "preparingEnvironment"
    | "installingCore"
    | "installingDependencies"
    | "buildingRosWorkspace"
    | "runningHealthChecks"
    | "recordingReceipt"
    | "completed"
    | "failed";

export interface RuntimeSetupSnapshot {
  schema_version: "dronedream.autonomy.runtime-setup.v1";
  operation_id: string | null;
  phase: RuntimeSetupPhase;
  progress: number;
  active: boolean;
  error: string | null;
  failed_phase: RuntimeSetupPhase | null;
  started_at: string | null;
  updated_at: string;
}
