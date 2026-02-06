use salvo::prelude::*;

use crate::types::{ApiError, ApiErrorDetail};

/// Render a standardized error response with proper HTTP status code
pub fn render_error(res: &mut Response, status: salvo::http::StatusCode, message: &str, error_type: &str) {
    res.status_code(status);
    res.render(Json(ApiError {
        error: ApiErrorDetail {
            message: message.to_string(),
            r#type: error_type.to_string(),
            code: None,
        },
    }));
}
