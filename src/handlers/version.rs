//! Version and capability endpoints.

use salvo::prelude::*;

use crate::version::{self, VersionResponse};

/// GET /v1/version — returns server capabilities and versions.
#[handler]
pub async fn get_version(res: &mut Response) {
    let registry = version::capability_registry();
    let response = VersionResponse {
        ominix_api: version::API_VERSION.to_string(),
        capabilities: registry,
        models_loaded: vec![], // TODO: populate from AppState when wired
    };
    res.render(Json(response));
}
