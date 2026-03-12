//! OpenAI API client implementation using the OpenAI-compatible base
//!
//! This module provides integration with OpenAI's GPT models through their API.
//! It supports both the Chat Completions API and the Responses API.

use crate::builder::{LLMBackend, LLMBuilder};
use crate::chat::Usage;
use crate::embedding::EmbeddingBuilder;
use crate::providers::openai_compatible::{
    OpenAIChatMessage, OpenAIChatResponse, OpenAICompatibleProvider, OpenAIProviderConfig,
    OpenAIResponseFormat, OpenAIStreamOptions, create_sse_stream,
};
use crate::{
    LLMProvider, ToolCall,
    chat::{
        ChatMessage, ChatProvider, ChatResponse, ChatRole, MessageType, StreamChunk,
        StreamResponse, StructuredOutputFormat, Tool, ToolChoice,
    },
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRequest, ModelListResponse, ModelsProvider, StandardModelListResponse},
};
use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

/// Provider-specific configuration for the OpenAI builder.
#[derive(Debug, Default, Clone)]
pub struct OpenAIConfig {
    pub voice: Option<String>,
}

/// Internal OpenAI provider config (for OpenAICompatibleProvider)
struct OpenAIInternalCfg;

impl OpenAIProviderConfig for OpenAIInternalCfg {
    const PROVIDER_NAME: &'static str = "OpenAI";
    const DEFAULT_BASE_URL: &'static str = "https://api.openai.com/v1/";
    const DEFAULT_MODEL: &'static str = "gpt-4.1-nano";
    const SUPPORTS_REASONING_EFFORT: bool = true;
    const SUPPORTS_STRUCTURED_OUTPUT: bool = true;
    const SUPPORTS_PARALLEL_TOOL_CALLS: bool = false;
    const SUPPORTS_STREAM_OPTIONS: bool = true;
}

// ---------------------------------------------------------------------------
// OpenAI struct
// ---------------------------------------------------------------------------

// NOTE: OpenAI cannot directly use the OpenAICompatibleProvider type alias, as it needs specific fields

/// Client for OpenAI API
pub struct OpenAI {
    // Delegate to the generic provider for common functionality
    provider: OpenAICompatibleProvider<OpenAIInternalCfg>,
    /// Whether to use the Responses API. None = auto-detect from model name.
    pub use_responses_api: Option<bool>,
    /// Previous response ID for Responses API conversation chaining.
    /// When set, the Responses API will use this to continue a conversation
    /// without resending the full message history (40-80% cache improvement).
    pub previous_response_id: Option<String>,
    pub enable_web_search: bool,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
}

// ---------------------------------------------------------------------------
// Shared types (used by both APIs)
// ---------------------------------------------------------------------------

/// OpenAI-specific tool that can be either a function tool or a built-in tool
#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum OpenAITool {
    Function {
        #[serde(rename = "type")]
        tool_type: String,
        function: crate::chat::FunctionTool,
    },
    /// Responses API function tool format: name/description/parameters at top level.
    ResponsesFunction {
        #[serde(rename = "type")]
        tool_type: String,
        name: String,
        description: String,
        parameters: serde_json::Value,
    },
    WebSearch {
        #[serde(rename = "type")]
        tool_type: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<UserLocation>,
    },
    FileSearch {
        #[serde(rename = "type")]
        tool_type: String,
        vector_store_ids: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_num_results: Option<u32>,
    },
    CodeInterpreter {
        #[serde(rename = "type")]
        tool_type: String,
    },
}

#[derive(Deserialize, Debug, Serialize)]
pub struct UserLocation {
    #[serde(rename = "type")]
    pub location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize)]
pub struct ApproximateLocation {
    pub country: String,
    pub city: String,
    pub region: String,
}

// ---------------------------------------------------------------------------
// Chat Completions API types
// ---------------------------------------------------------------------------

/// Response for chat with web search (legacy partial Responses API usage)
#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchChatResponse {
    pub output: Vec<OpenAIWebSearchOutput>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchOutput {
    pub content: Option<Vec<OpenAIWebSearchContent>>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIWebSearchContent {
    #[serde(rename = "type")]
    pub msg_type: String,
    pub text: String,
}

impl std::fmt::Display for OpenAIWebSearchChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = self.text() {
            write!(f, "{text}")
        } else {
            write!(f, "No response content")
        }
    }
}

impl ChatResponse for OpenAIWebSearchChatResponse {
    fn text(&self) -> Option<String> {
        self.output
            .last()
            .and_then(|output| output.content.as_ref())
            .and_then(|content| content.last())
            .map(|content| content.text.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        None // Web search responses don't contain tool calls
    }

    fn thinking(&self) -> Option<String> {
        None
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

/// Request payload for OpenAI's chat API endpoint.
#[derive(Serialize, Debug)]
pub struct OpenAIAPIChatRequest<'a> {
    pub model: &'a str,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<OpenAIChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<OpenAIStreamOptions>,
    #[serde(flatten)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Responses API types
// ---------------------------------------------------------------------------

/// Request payload for OpenAI's Responses API endpoint (POST /v1/responses).
#[derive(Serialize, Debug)]
pub struct OpenAIResponsesRequest<'a> {
    pub model: &'a str,
    pub input: ResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextFormat>,
    pub stream: bool,
    #[serde(flatten)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

/// Input can be a simple string or array of items
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<InputItem>),
}

/// Input item types following the Responses API spec
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum InputItem {
    #[serde(rename = "message")]
    Message {
        role: String,
        content: MessageContent,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput { call_id: String, output: String },
}

/// Message content: string or array of content parts
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// Content part types within a message
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "input_text")]
    InputText { text: String },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(flatten)]
        source: ImageSource,
    },
}

/// Image source: URL or base64
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ImageSource {
    Url { url: String },
    Base64 { data: String, media_type: String },
}

/// Structured output format for the Responses API `text` parameter
#[derive(Debug, Serialize)]
pub struct ResponsesTextFormat {
    pub format: ResponsesTextFormatType,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesTextFormatType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "json_schema")]
    JsonSchema {
        name: String,
        schema: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

/// Reasoning configuration for the Responses API
#[derive(Debug, Serialize)]
pub struct ResponsesReasoningConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl From<StructuredOutputFormat> for ResponsesTextFormat {
    fn from(sof: StructuredOutputFormat) -> Self {
        match sof.schema {
            Some(mut schema) => {
                if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                }
                ResponsesTextFormat {
                    format: ResponsesTextFormatType::JsonSchema {
                        name: sof.name,
                        schema,
                        strict: sof.strict,
                    },
                }
            }
            None => ResponsesTextFormat {
                format: ResponsesTextFormatType::JsonSchema {
                    name: sof.name,
                    schema: serde_json::json!({}),
                    strict: sof.strict,
                },
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Responses API response types
// ---------------------------------------------------------------------------

/// A part of a reasoning summary
#[derive(Debug, Clone, Deserialize)]
pub struct ReasoningSummaryPart {
    #[serde(rename = "type")]
    pub part_type: Option<String>,
    pub text: Option<String>,
}

/// Full response from POST /v1/responses
#[derive(Debug, Deserialize)]
pub struct OpenAIResponsesResponse {
    pub id: String,
    pub status: String,
    pub output: Vec<OutputItem>,
    pub usage: Option<Usage>,
    #[serde(default)]
    pub model: Option<String>,
}

/// Output item types
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputItem {
    #[serde(rename = "message")]
    Message {
        id: String,
        role: String,
        content: Vec<OutputContentPart>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        id: String,
        #[serde(default)]
        summary: Vec<ReasoningSummaryPart>,
    },
    #[serde(rename = "web_search_call")]
    WebSearchCall { id: String, status: String },
    #[serde(rename = "file_search_call")]
    FileSearchCall { id: String, status: String },
    /// Catch-all for unknown output item types (forward compatibility)
    #[serde(other)]
    Unknown,
}

/// Output content part (within a message output item)
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum OutputContentPart {
    #[serde(rename = "output_text")]
    Text {
        text: String,
        #[serde(default)]
        annotations: Vec<Annotation>,
    },
    #[serde(rename = "refusal")]
    Refusal { refusal: String },
    #[serde(other)]
    Unknown,
}

/// Annotation on output text (e.g. web search citation, file citation)
#[derive(Debug, Clone, Deserialize)]
pub struct Annotation {
    #[serde(rename = "type")]
    pub annotation_type: Option<String>,
    /// Start index in the text
    pub start_index: Option<usize>,
    /// End index in the text
    pub end_index: Option<usize>,
    /// The URL for url_citation annotations
    pub url: Option<String>,
    /// The title for url_citation annotations
    pub title: Option<String>,
    /// File citation ID for file_citation annotations
    pub file_id: Option<String>,
}

impl std::fmt::Display for OpenAIResponsesResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(text) = self.text() {
            write!(f, "{text}")
        } else {
            write!(f, "No response content")
        }
    }
}

impl ChatResponse for OpenAIResponsesResponse {
    fn text(&self) -> Option<String> {
        let mut text = String::new();
        for item in &self.output {
            if let OutputItem::Message { content, .. } = item {
                for part in content {
                    if let OutputContentPart::Text { text: t, .. } = part {
                        text.push_str(t);
                    }
                }
            }
        }
        if text.is_empty() { None } else { Some(text) }
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        let calls: Vec<ToolCall> = self
            .output
            .iter()
            .filter_map(|item| {
                if let OutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } = item
                {
                    Some(ToolCall {
                        id: call_id.clone(),
                        call_type: "function".into(),
                        function: crate::FunctionCall {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect();
        if calls.is_empty() { None } else { Some(calls) }
    }

    fn thinking(&self) -> Option<String> {
        let mut thinking = String::new();
        for item in &self.output {
            if let OutputItem::Reasoning { summary, .. } = item {
                for part in summary {
                    if let Some(text) = &part.text {
                        if !thinking.is_empty() {
                            thinking.push('\n');
                        }
                        thinking.push_str(text);
                    }
                }
            }
        }
        if thinking.is_empty() {
            None
        } else {
            Some(thinking)
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }

    fn response_id(&self) -> Option<&str> {
        Some(&self.id)
    }
}

// ---------------------------------------------------------------------------
// Responses API streaming types
// ---------------------------------------------------------------------------

#[allow(dead_code)]
struct ResponsesItemState {
    id: String,
    item_type: String,
    name: Option<String>,
    call_id: Option<String>,
    output_index: usize,
}

struct FnCallState {
    call_id: String,
    name: String,
    arguments_buffer: String,
    output_index: usize,
}

/// State machine for parsing Responses API streaming events.
struct ResponsesStreamParser {
    items: HashMap<usize, ResponsesItemState>,
    fn_arg_buffers: HashMap<String, FnCallState>,
    has_function_calls: bool,
    pending: Vec<Result<StreamChunk, LLMError>>,
}

impl ResponsesStreamParser {
    fn new() -> Self {
        Self {
            items: HashMap::new(),
            fn_arg_buffers: HashMap::new(),
            has_function_calls: false,
            pending: Vec::new(),
        }
    }

    fn process_event(
        &mut self,
        event_type: &str,
        data: &str,
    ) -> Vec<Result<StreamChunk, LLMError>> {
        match event_type {
            "response.output_item.added" => self.handle_item_added(data),
            "response.output_text.delta" => self.handle_text_delta(data),
            "response.function_call_arguments.delta" => self.handle_fn_args_delta(data),
            "response.function_call_arguments.done" => self.handle_fn_args_done(data),
            "response.completed" => self.handle_completed(data),
            "response.failed" => self.handle_failed(data),
            "response.incomplete" => self.handle_incomplete(),
            _ => {} // Ignore unknown events
        }
        std::mem::take(&mut self.pending)
    }

    fn handle_item_added(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct ItemAddedEvent {
            output_index: usize,
            item: ItemAddedItem,
        }
        #[derive(Deserialize)]
        struct ItemAddedItem {
            id: String,
            #[serde(rename = "type")]
            item_type: String,
            #[serde(default)]
            name: Option<String>,
            #[serde(default)]
            call_id: Option<String>,
        }

        let Ok(event) = serde_json::from_str::<ItemAddedEvent>(data) else {
            log::debug!("Failed to parse output_item.added event: {}", data);
            return;
        };

        let idx = event.output_index;
        let item = event.item;

        if item.item_type == "function_call" {
            self.has_function_calls = true;
            let name = item.name.clone().unwrap_or_default();
            let call_id = item.call_id.clone().unwrap_or_default();

            self.pending.push(Ok(StreamChunk::ToolUseStart {
                index: idx,
                id: call_id.clone(),
                name: name.clone(),
            }));

            self.fn_arg_buffers.insert(
                item.id.clone(),
                FnCallState {
                    call_id,
                    name,
                    arguments_buffer: String::new(),
                    output_index: idx,
                },
            );
        }

        self.items.insert(
            idx,
            ResponsesItemState {
                id: item.id,
                item_type: item.item_type,
                name: item.name,
                call_id: item.call_id,
                output_index: idx,
            },
        );
    }

    fn handle_text_delta(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct TextDeltaEvent {
            delta: String,
        }

        let Ok(event) = serde_json::from_str::<TextDeltaEvent>(data) else {
            log::debug!("Failed to parse output_text.delta event: {}", data);
            return;
        };
        if !event.delta.is_empty() {
            self.pending.push(Ok(StreamChunk::Text(event.delta)));
        }
    }

    fn handle_fn_args_delta(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct FnArgsDeltaEvent {
            item_id: String,
            delta: String,
        }

        let Ok(event) = serde_json::from_str::<FnArgsDeltaEvent>(data) else {
            log::debug!(
                "Failed to parse function_call_arguments.delta event: {}",
                data
            );
            return;
        };

        if let Some(state) = self.fn_arg_buffers.get_mut(&event.item_id) {
            state.arguments_buffer.push_str(&event.delta);
            self.pending.push(Ok(StreamChunk::ToolUseInputDelta {
                index: state.output_index,
                partial_json: event.delta,
            }));
        }
    }

    fn handle_fn_args_done(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct FnArgsDoneEvent {
            item_id: String,
            arguments: String,
        }

        let Ok(event) = serde_json::from_str::<FnArgsDoneEvent>(data) else {
            log::debug!(
                "Failed to parse function_call_arguments.done event: {}",
                data
            );
            return;
        };

        if let Some(state) = self.fn_arg_buffers.remove(&event.item_id) {
            self.pending.push(Ok(StreamChunk::ToolUseComplete {
                index: state.output_index,
                tool_call: ToolCall {
                    id: state.call_id,
                    call_type: "function".into(),
                    function: crate::FunctionCall {
                        name: state.name,
                        arguments: event.arguments,
                    },
                },
            }));
        }
    }

    fn handle_completed(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct CompletedEvent {
            response: CompletedResponse,
        }
        #[derive(Deserialize)]
        struct CompletedResponse {
            usage: Option<Usage>,
        }

        let usage = serde_json::from_str::<CompletedEvent>(data)
            .ok()
            .and_then(|e| e.response.usage);

        if let Some(usage) = usage {
            self.pending.push(Ok(StreamChunk::Usage(usage)));
        }

        let stop_reason = if self.has_function_calls {
            "tool_use"
        } else {
            "end_turn"
        };
        self.pending.push(Ok(StreamChunk::Done {
            stop_reason: stop_reason.to_string(),
        }));
    }

    fn handle_failed(&mut self, data: &str) {
        #[derive(Deserialize)]
        struct FailedEvent {
            response: Option<FailedResponse>,
        }
        #[derive(Deserialize)]
        struct FailedResponse {
            error: Option<FailedError>,
        }
        #[derive(Deserialize)]
        struct FailedError {
            message: Option<String>,
        }

        let msg = serde_json::from_str::<FailedEvent>(data)
            .ok()
            .and_then(|e| e.response)
            .and_then(|r| r.error)
            .and_then(|e| e.message)
            .unwrap_or_else(|| "Response generation failed".to_string());

        self.pending.push(Err(LLMError::ProviderError(msg)));
    }

    fn handle_incomplete(&mut self) {
        self.pending.push(Ok(StreamChunk::Done {
            stop_reason: "max_tokens".to_string(),
        }));
    }
}

// ---------------------------------------------------------------------------
// OpenAI constructor & helpers
// ---------------------------------------------------------------------------

impl OpenAI {
    /// Model families that strongly prefer or require the Responses API.
    const RESPONSES_API_MODEL_FAMILIES: &[&str] = &["o1", "o3", "o4", "gpt-5"];
    /// Error markers indicating Responses API is unavailable for the current backend.
    const RESPONSES_API_FALLBACK_MARKERS: &[&str] = &[
        "responses api is enabled only for api-version",
        "api-version 2025-03-01-preview",
        "responses api is not supported",
        "does not support the responses api",
        "not supported in v1/responses",
        "v1/responses is not supported",
    ];

    fn model_has_family(model: &str, family: &str) -> bool {
        model == family
            || model.starts_with(&format!("{family}-"))
            || model.starts_with(&format!("{family}."))
            || model.contains(&format!("/{family}"))
            || model.contains(&format!(":{family}"))
    }

    /// Creates a new OpenAI client with the specified configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tool_choice: Option<ToolChoice>,
        normalize_response: Option<bool>,
        reasoning_effort: Option<String>,
        voice: Option<String>,
        extra_body: Option<serde_json::Value>,
        enable_web_search: Option<bool>,
        web_search_context_size: Option<String>,
        web_search_user_location_type: Option<String>,
        web_search_user_location_approximate_country: Option<String>,
        web_search_user_location_approximate_city: Option<String>,
        web_search_user_location_approximate_region: Option<String>,
    ) -> Result<Self, LLMError> {
        let api_key_str = api_key.into();
        if api_key_str.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }
        Ok(OpenAI {
            provider: <OpenAICompatibleProvider<OpenAIInternalCfg>>::new(
                api_key_str,
                base_url,
                model,
                max_tokens,
                temperature,
                timeout_seconds,
                top_p,
                top_k,
                tool_choice,
                reasoning_effort,
                voice,
                extra_body,
                None, // parallel_tool_calls
                normalize_response,
                embedding_encoding_format,
                embedding_dimensions,
            ),
            use_responses_api: None, // auto-detect from model
            previous_response_id: None,
            enable_web_search: enable_web_search.unwrap_or(false),
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        })
    }

    fn should_use_responses_api(&self) -> bool {
        match self.use_responses_api {
            Some(true) => true,
            Some(false) => false,
            None => {
                let model = self.provider.model.trim().to_lowercase();

                // Codex-family models consistently use the Responses API.
                if model.contains("codex") {
                    return true;
                }

                Self::RESPONSES_API_MODEL_FAMILIES
                    .iter()
                    .any(|family| Self::model_has_family(&model, family))
            }
        }
    }

    fn should_fallback_from_responses_api_error(&self, error: &LLMError) -> bool {
        if self.use_responses_api == Some(true) {
            return false;
        }

        let haystack = match error {
            LLMError::ResponseFormatError {
                message,
                raw_response,
            } => format!("{message}\n{raw_response}"),
            LLMError::ProviderError(message) | LLMError::HttpError(message) => message.clone(),
            _ => return false,
        };

        let lower = haystack.to_ascii_lowercase();
        let is_responses_related = lower.contains("responses api")
            || lower.contains("v1/responses")
            || lower.contains("/responses");
        if !is_responses_related {
            return false;
        }

        Self::RESPONSES_API_FALLBACK_MARKERS
            .iter()
            .any(|marker| lower.contains(marker))
    }
}

impl crate::HasConfig for OpenAI {
    type Config = OpenAIConfig;
}


// Helper methods to access provider fields
impl OpenAI {
    pub fn api_key(&self) -> &str {
        &self.provider.api_key
    }

    pub fn model(&self) -> &str {
        &self.provider.model
    }

    pub fn base_url(&self) -> &reqwest::Url {
        &self.provider.base_url
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.provider.timeout_seconds
    }

    pub fn client(&self) -> &reqwest::Client {
        &self.provider.client
    }

    /// Set the previous response ID for conversation chaining.
    /// When set, subsequent Responses API calls will include this ID,
    /// enabling 40-80% cache utilization improvement.
    pub fn set_previous_response_id(&mut self, id: Option<String>) {
        self.previous_response_id = id;
    }

    /// Update `previous_response_id` from a `ChatResponse`.
    /// Call this after each successful Responses API call to automatically
    /// chain conversations:
    ///
    /// ```ignore
    /// let response = openai.chat(&messages, None).await?;
    /// openai.chain_response(&*response);
    /// // Next call will use previous_response_id automatically
    /// let follow_up = openai.chat(&more_messages, None).await?;
    /// ```
    pub fn chain_response(&mut self, response: &dyn ChatResponse) {
        self.previous_response_id = response.response_id().map(String::from);
    }

    fn build_function_tools(&self, tools: Option<&[Tool]>) -> Option<Vec<OpenAITool>> {
        let mut openai_tools: Vec<OpenAITool> = Vec::new();
        if let Some(tools) = tools {
            for tool in tools {
                openai_tools.push(OpenAITool::Function {
                    tool_type: tool.tool_type.clone(),
                    function: tool.function.clone(),
                });
            }
        }
        if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        }
    }

    /// Build tools in the Responses API format where name/description/parameters
    /// are top-level fields instead of nested under `function`.
    fn build_responses_function_tools(&self, tools: Option<&[Tool]>) -> Option<Vec<OpenAITool>> {
        let mut openai_tools: Vec<OpenAITool> = Vec::new();
        if let Some(tools) = tools {
            for tool in tools {
                openai_tools.push(OpenAITool::ResponsesFunction {
                    tool_type: "function".to_string(),
                    name: tool.function.name.clone(),
                    description: tool.function.description.clone(),
                    parameters: tool.function.parameters.clone(),
                });
            }
        }
        if openai_tools.is_empty() {
            None
        } else {
            Some(openai_tools)
        }
    }

    fn resolve_tool_choice_for_request(
        &self,
        tools: &Option<Vec<OpenAITool>>,
    ) -> Option<ToolChoice> {
        if tools.is_some() {
            self.provider.tool_choice.clone()
        } else {
            None
        }
    }

    fn build_web_search_tool(&self) -> OpenAITool {
        let loc_type_opt = self
            .web_search_user_location_type
            .as_ref()
            .filter(|t| matches!(t.as_str(), "exact" | "approximate"));
        let country = self.web_search_user_location_approximate_country.as_ref();
        let city = self.web_search_user_location_approximate_city.as_ref();
        let region = self.web_search_user_location_approximate_region.as_ref();
        let approximate = if [country, city, region].iter().any(|v| v.is_some()) {
            Some(ApproximateLocation {
                country: country.cloned().unwrap_or_default(),
                city: city.cloned().unwrap_or_default(),
                region: region.cloned().unwrap_or_default(),
            })
        } else {
            None
        };
        let user_location = loc_type_opt.map(|loc_type| UserLocation {
            location_type: loc_type.clone(),
            approximate,
        });
        OpenAITool::WebSearch {
            tool_type: "web_search_preview".to_string(),
            user_location,
        }
    }
}

// ---------------------------------------------------------------------------
// Responses API helpers on OpenAI
// ---------------------------------------------------------------------------

impl OpenAI {
    fn responses_chunk_to_stream_response(
        result: Result<StreamChunk, LLMError>,
    ) -> Option<Result<StreamResponse, LLMError>> {
        match result {
            Ok(StreamChunk::Text(text)) => Some(Ok(StreamResponse {
                choices: vec![crate::chat::StreamChoice {
                    delta: crate::chat::StreamDelta {
                        content: Some(text),
                        reasoning_content: None,
                        tool_calls: None,
                    },
                }],
                usage: None,
            })),
            Ok(StreamChunk::ToolUseComplete { tool_call, .. }) => Some(Ok(StreamResponse {
                choices: vec![crate::chat::StreamChoice {
                    delta: crate::chat::StreamDelta {
                        content: None,
                        reasoning_content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                }],
                usage: None,
            })),
            Ok(StreamChunk::Usage(usage)) => Some(Ok(StreamResponse {
                choices: vec![],
                usage: Some(usage),
            })),
            Err(e) => Some(Err(e)),
            _ => None,
        }
    }

    /// Convert ChatMessage slice to Responses API input items.
    /// System messages are extracted as `instructions`.
    fn prepare_responses_input(
        &self,
        messages: &[ChatMessage],
    ) -> (Option<String>, Vec<InputItem>) {
        let mut instructions = None;
        let mut items = Vec::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    instructions = Some(msg.content.clone());
                }
                ChatRole::User => {
                    let content = match &msg.message_type {
                        MessageType::Image((mime, data)) => {
                            let mut parts = Vec::new();
                            if !msg.content.is_empty() {
                                parts.push(ContentPart::InputText {
                                    text: msg.content.clone(),
                                });
                            }
                            parts.push(ContentPart::InputImage {
                                source: ImageSource::Base64 {
                                    data: BASE64.encode(data),
                                    media_type: mime.mime_type().to_string(),
                                },
                            });
                            MessageContent::Parts(parts)
                        }
                        MessageType::ImageURL(url) => {
                            let mut parts = Vec::new();
                            if !msg.content.is_empty() {
                                parts.push(ContentPart::InputText {
                                    text: msg.content.clone(),
                                });
                            }
                            parts.push(ContentPart::InputImage {
                                source: ImageSource::Url { url: url.clone() },
                            });
                            MessageContent::Parts(parts)
                        }
                        _ => MessageContent::Text(msg.content.clone()),
                    };
                    items.push(InputItem::Message {
                        role: "user".into(),
                        content,
                    });
                }
                ChatRole::Assistant => {
                    if let MessageType::ToolUse(tool_calls) = &msg.message_type {
                        for tc in tool_calls {
                            items.push(InputItem::FunctionCall {
                                call_id: tc.id.clone(),
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            });
                        }
                    } else {
                        items.push(InputItem::Message {
                            role: "assistant".into(),
                            content: MessageContent::Text(msg.content.clone()),
                        });
                    }
                }
                ChatRole::Tool => {
                    if let MessageType::ToolResult(tool_calls) = &msg.message_type {
                        for tc in tool_calls {
                            items.push(InputItem::FunctionCallOutput {
                                call_id: tc.id.clone(),
                                output: if !msg.content.is_empty() {
                                    msg.content.clone()
                                } else {
                                    tc.function.arguments.clone()
                                },
                            });
                        }
                    }
                }
            }
        }

        (instructions, items)
    }

    /// Send a request to the Responses API and return the response.
    async fn send_responses_request(
        &self,
        body: &OpenAIResponsesRequest<'_>,
    ) -> Result<reqwest::Response, LLMError> {
        let url = self
            .provider
            .base_url
            .join("responses")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(body)
        {
            log::trace!("OpenAI Responses API request payload: {}", json);
        }

        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(body);

        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;
        log::debug!("OpenAI Responses API HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI Responses API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(response)
    }

    /// Non-streaming chat with tools via Responses API
    async fn chat_with_tools_responses(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let (instructions, items) = self.prepare_responses_input(messages);
        let final_tools = self.build_responses_function_tools(tools);
        let request_tool_choice = self.resolve_tool_choice_for_request(&final_tools);
        let text_format = json_schema.map(ResponsesTextFormat::from);
        let reasoning =
            self.provider
                .reasoning_effort
                .as_ref()
                .map(|effort| ResponsesReasoningConfig {
                    effort: Some(effort.clone()),
                    summary: Some("auto".into()),
                });

        let body = OpenAIResponsesRequest {
            model: &self.provider.model,
            input: ResponsesInput::Items(items),
            instructions: instructions.as_deref(),
            tools: final_tools,
            tool_choice: request_tool_choice,
            temperature: self.provider.temperature,
            top_p: self.provider.top_p,
            max_output_tokens: self.provider.max_tokens,
            previous_response_id: self.previous_response_id.as_deref(),
            reasoning,
            text: text_format,
            stream: false,
            extra_body: self.provider.extra_body.clone(),
        };

        let response = self.send_responses_request(&body).await?;
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIResponsesResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI Responses API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    /// Streaming chat with tools via Responses API
    async fn chat_stream_with_tools_responses(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        let (instructions, items) = self.prepare_responses_input(messages);
        let final_tools = self.build_responses_function_tools(tools);
        let request_tool_choice = self.resolve_tool_choice_for_request(&final_tools);
        let text_format = json_schema.map(ResponsesTextFormat::from);
        let reasoning =
            self.provider
                .reasoning_effort
                .as_ref()
                .map(|effort| ResponsesReasoningConfig {
                    effort: Some(effort.clone()),
                    summary: Some("auto".into()),
                });

        let body = OpenAIResponsesRequest {
            model: &self.provider.model,
            input: ResponsesInput::Items(items),
            instructions: instructions.as_deref(),
            tools: final_tools,
            tool_choice: request_tool_choice,
            temperature: self.provider.temperature,
            top_p: self.provider.top_p,
            max_output_tokens: self.provider.max_tokens,
            previous_response_id: self.previous_response_id.as_deref(),
            reasoning,
            text: text_format,
            stream: true,
            extra_body: self.provider.extra_body.clone(),
        };

        let response = self.send_responses_request(&body).await?;

        let stream = response.bytes_stream();
        let parser_stream = futures::stream::unfold(
            (stream, ResponsesStreamParser::new(), String::new()),
            |(mut stream, mut parser, mut buffer)| async move {
                loop {
                    // First try to parse any complete events from the buffer
                    while let Some(event_end) = buffer.find("\n\n") {
                        let event_block = buffer[..event_end].to_string();
                        buffer = buffer[event_end + 2..].to_string();

                        let mut event_type = String::new();
                        let mut event_data = String::new();

                        for line in event_block.lines() {
                            if let Some(rest) = line.strip_prefix("event: ") {
                                event_type = rest.trim().to_string();
                            } else if let Some(rest) = line.strip_prefix("data: ") {
                                if !event_data.is_empty() {
                                    event_data.push('\n');
                                }
                                event_data.push_str(rest.trim());
                            } else if let Some(rest) = line.strip_prefix("event:") {
                                event_type = rest.trim().to_string();
                            } else if let Some(rest) = line.strip_prefix("data:") {
                                if !event_data.is_empty() {
                                    event_data.push('\n');
                                }
                                event_data.push_str(rest.trim());
                            }
                        }

                        if !event_type.is_empty() {
                            let chunks = parser.process_event(&event_type, &event_data);
                            if !chunks.is_empty() {
                                return Some((
                                    futures::stream::iter(chunks),
                                    (stream, parser, buffer),
                                ));
                            }
                        }
                    }

                    // Need more data from the network
                    match stream.next().await {
                        Some(Ok(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Some(Err(e)) => {
                            return Some((
                                futures::stream::iter(vec![Err(LLMError::HttpError(
                                    e.to_string(),
                                ))]),
                                (stream, parser, buffer),
                            ));
                        }
                        None => {
                            // Stream ended
                            return None;
                        }
                    }
                }
            },
        )
        .flatten();

        Ok(Box::pin(parser_stream))
    }

    /// Streaming chat with structured responses via Responses API
    async fn chat_stream_struct_responses(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        let chunk_stream = self
            .chat_stream_with_tools_responses(messages, tools, json_schema)
            .await?;

        let struct_stream = chunk_stream
            .filter_map(|result| async move { OpenAI::responses_chunk_to_stream_response(result) });

        Ok(Box::pin(struct_stream))
    }

    /// Chat with OpenAI-hosted tools using the `/responses` endpoint
    pub async fn chat_with_hosted_tools(
        &self,
        input: String,
        hosted_tools: Vec<OpenAITool>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let body = OpenAIAPIChatRequest {
            model: self.provider.model.as_str(),
            messages: Vec::new(),
            input: Some(input),
            max_completion_tokens: None,
            max_output_tokens: self.provider.max_tokens,
            temperature: self.provider.temperature,
            stream: false,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: Some(hosted_tools),
            tool_choice: self.provider.tool_choice.clone(),
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format: None,
            stream_options: None,
            extra_body: self.provider.extra_body.clone(),
        };

        let url = self
            .provider
            .base_url
            .join("responses")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);

        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&body)
        {
            log::trace!("OpenAI hosted tools request payload: {}", json);
        }

        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;
        log::debug!("OpenAI hosted tools HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI hosted tools API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIWebSearchChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI hosted tools API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// ChatProvider trait implementation
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

#[async_trait]
impl ChatProvider for OpenAI {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.should_use_responses_api() {
            match self
                .chat_with_tools_responses(messages, tools, json_schema.clone())
                .await
            {
                Ok(response) => return Ok(response),
                Err(error) if self.should_fallback_from_responses_api_error(&error) => {
                    log::warn!(
                        "OpenAI Responses API unavailable for model '{}', falling back to chat/completions: {}",
                        self.provider.model,
                        error
                    );
                }
                Err(error) => return Err(error),
            }
        }

        // Chat Completions path (existing)
        let openai_msgs = self.provider.prepare_messages(messages);
        let response_format: Option<OpenAIResponseFormat> = json_schema.clone().map(|s| s.into());
        let final_tools = self.build_function_tools(tools);
        let request_tool_choice = self.resolve_tool_choice_for_request(&final_tools);
        let body = OpenAIAPIChatRequest {
            model: self.provider.model.as_str(),
            messages: openai_msgs,
            input: None,
            max_completion_tokens: self.provider.max_tokens,
            max_output_tokens: None,
            temperature: self.provider.temperature,
            stream: false,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: final_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format,
            stream_options: None,
            extra_body: self.provider.extra_body.clone(),
        };
        let url = self
            .provider
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);
        if log::log_enabled!(log::Level::Trace)
            && let Ok(json) = serde_json::to_string(&body)
        {
            log::trace!("OpenAI request payload: {}", json);
        }
        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("OpenAI HTTP status: {}", response.status());
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat_with_web_search(&self, input: String) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.should_use_responses_api() {
            // Use the Responses API natively with the web_search_preview tool
            let messages = vec![ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: input,
            }];
            let web_search_tool = self.build_web_search_tool();
            let (instructions, items) = self.prepare_responses_input(&messages);
            let reasoning =
                self.provider
                    .reasoning_effort
                    .as_ref()
                    .map(|effort| ResponsesReasoningConfig {
                        effort: Some(effort.clone()),
                        summary: Some("auto".into()),
                    });
            let body = OpenAIResponsesRequest {
                model: &self.provider.model,
                input: ResponsesInput::Items(items),
                instructions: instructions.as_deref(),
                tools: Some(vec![web_search_tool]),
                tool_choice: None,
                temperature: self.provider.temperature,
                top_p: self.provider.top_p,
                max_output_tokens: self.provider.max_tokens,
                previous_response_id: self.previous_response_id.as_deref(),
                reasoning,
                text: None,
                stream: false,
                extra_body: self.provider.extra_body.clone(),
            };
            let response = self.send_responses_request(&body).await?;
            let resp_text = response.text().await?;
            let json_resp: Result<OpenAIResponsesResponse, serde_json::Error> =
                serde_json::from_str(&resp_text);
            return match json_resp {
                Ok(response) => Ok(Box::new(response)),
                Err(e) => Err(LLMError::ResponseFormatError {
                    message: format!("Failed to decode OpenAI Responses API response: {e}"),
                    raw_response: resp_text,
                }),
            };
        }

        // Legacy path using chat_with_hosted_tools
        let web_search_tool = self.build_web_search_tool();
        self.chat_with_hosted_tools(input, vec![web_search_tool])
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        if self.should_use_responses_api() {
            let chunk_stream = self
                .chat_stream_with_tools_responses(messages, None, json_schema)
                .await?;
            let text_stream = chunk_stream.filter_map(|result| async move {
                match result {
                    Ok(StreamChunk::Text(text)) if !text.is_empty() => Some(Ok(text)),
                    Err(e) => Some(Err(e)),
                    _ => None,
                }
            });
            return Ok(Box::pin(text_stream));
        }

        let struct_stream = self.chat_stream_struct(messages, None, json_schema).await?;
        let content_stream = struct_stream.filter_map(|result| async move {
            match result {
                Ok(stream_response) => {
                    if let Some(choice) = stream_response.choices.first()
                        && let Some(content) = &choice.delta.content
                        && !content.is_empty()
                    {
                        return Some(Ok(content.clone()));
                    }
                    None
                }
                Err(e) => Some(Err(e)),
            }
        });
        Ok(Box::pin(content_stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        if self.should_use_responses_api() {
            return self
                .chat_stream_struct_responses(messages, tools, json_schema)
                .await;
        }

        let openai_msgs = self.provider.prepare_messages(messages);
        let openai_tools = self.build_function_tools(tools);
        let repsonse_schema: Option<OpenAIResponseFormat> = json_schema.map(|schema| schema.into());
        let body = OpenAIAPIChatRequest {
            model: &self.provider.model,
            messages: openai_msgs,
            input: None,
            max_completion_tokens: self.provider.max_tokens,
            max_output_tokens: None,
            temperature: self.provider.temperature,
            stream: true,
            top_p: self.provider.top_p,
            top_k: self.provider.top_k,
            tools: openai_tools,
            tool_choice: self.provider.tool_choice.clone(),
            reasoning_effort: self.provider.reasoning_effort.clone(),
            response_format: repsonse_schema,
            stream_options: Some(OpenAIStreamOptions {
                include_usage: true,
            }),
            extra_body: self.provider.extra_body.clone(),
        };
        let url = self
            .provider
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body);
        if let Some(timeout) = self.provider.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(create_sse_stream(
            response,
            self.provider.normalize_response,
        ))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        if self.should_use_responses_api() {
            return self
                .chat_stream_with_tools_responses(messages, tools, json_schema)
                .await;
        }

        // Delegate to the inner OpenAICompatibleProvider which has the full implementation
        self.provider
            .chat_stream_with_tools(messages, tools, json_schema)
            .await
    }
}

// ---------------------------------------------------------------------------
// Other trait implementations
// ---------------------------------------------------------------------------

#[async_trait]
impl CompletionProvider for OpenAI {
    async fn complete(
        &self,
        _req: &CompletionRequest,
        _json_schema: Option<StructuredOutputFormat>,
    ) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "OpenAI completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl ModelsProvider for OpenAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url()
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client()
            .get(url)
            .bearer_auth(self.api_key())
            .send()
            .await?
            .error_for_status()?;

        let result = StandardModelListResponse {
            inner: resp.json().await?,
            backend: LLMBackend::OpenAI,
        };
        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

impl LLMBuilder<OpenAI> {
    /// Set the voice.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.config.voice = Some(voice.into());
        self
    }

    pub fn build(self) -> Result<Arc<OpenAI>, LLMError> {
        let key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for OpenAI".to_string())
        })?;
        let mut openai = OpenAI::new(
            key,
            self.base_url,
            self.model,
            self.max_tokens,
            self.temperature,
            self.timeout_seconds,
            self.top_p,
            self.top_k,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            self.tool_choice,
            self.normalize_response,
            self.reasoning_effort,
            self.config.voice,
            self.extra_body,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        openai.use_responses_api = self.use_responses_api;

        Ok(Arc::new(openai))
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.provider.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let emb_format = self
            .provider
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = OpenAIEmbeddingRequest {
            model: self.provider.model.to_string(),
            input,
            encoding_format: Some(emb_format),
            dimensions: self.provider.embedding_dimensions,
        };

        let url = self
            .provider
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

impl EmbeddingBuilder<OpenAI> {
    /// Build an OpenAI embedding provider.
    pub fn build(self) -> Result<Arc<OpenAI>, LLMError> {
        let api_key = self.api_key.ok_or_else(|| {
            LLMError::InvalidRequest("No API key provided for OpenAI".to_string())
        })?;

        let model = self
            .model
            .unwrap_or_else(|| "text-embedding-3-small".to_string());

        let provider = OpenAI::new(
            api_key,
            self.base_url,
            Some(model),
            None,
            None,
            self.timeout_seconds,
            None,
            None,
            self.embedding_encoding_format,
            self.embedding_dimensions,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;

        Ok(Arc::new(provider))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::LLMBuilder;
    use crate::chat::{FunctionTool, ImageMime, ToolChoice};
    use either::Either::Right;
    use serde_json::json;

    fn make_provider(model: &str) -> OpenAI {
        OpenAI::new(
            "key",
            None,
            Some(model.to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn make_provider_with_responses_api(model: &str, use_responses_api: Option<bool>) -> OpenAI {
        let mut p = OpenAI::new(
            "key",
            None,
            Some(model.to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        p.use_responses_api = use_responses_api;
        p
    }

    // -- should_use_responses_api tests --

    #[test]
    fn test_should_use_responses_api_auto_detect_codex() {
        let p = make_provider("codex-mini");
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_auto_detect_gpt4() {
        let p = make_provider("gpt-4.1");
        assert!(!p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_auto_detect_gpt5() {
        let p = make_provider("gpt-5-mini");
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_auto_detect_o1() {
        let p = make_provider("o1-mini");
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_auto_detect_prefixed_model() {
        let p = make_provider("openai/gpt-5");
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_auto_detect_trims_model_name() {
        let p = make_provider("  gpt-5  ");
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_explicit_true() {
        let p = make_provider_with_responses_api("gpt-4.1", Some(true));
        assert!(p.should_use_responses_api());
    }

    #[test]
    fn test_should_use_responses_api_explicit_false() {
        let p = make_provider_with_responses_api("codex-mini", Some(false));
        assert!(!p.should_use_responses_api());
    }

    #[test]
    fn test_should_fallback_from_responses_api_error_azure_version_gate() {
        let p = make_provider_with_responses_api("us/azure/openai/o3-mini", None);
        let error = LLMError::ResponseFormatError {
            message: "OpenAI Responses API returned error status: 400 Bad Request".to_string(),
            raw_response: r#"{"error":{"message":"Azure OpenAI Responses API is enabled only for api-version 2025-03-01-preview and later"}}"#.to_string(),
        };
        assert!(p.should_fallback_from_responses_api_error(&error));
    }

    #[test]
    fn test_should_not_fallback_from_responses_api_error_when_forced() {
        let p = make_provider_with_responses_api("o3-mini", Some(true));
        let error = LLMError::ResponseFormatError {
            message: "OpenAI Responses API returned error status: 400 Bad Request".to_string(),
            raw_response: r#"{"error":{"message":"Responses API is enabled only for api-version 2025-03-01-preview and later"}}"#.to_string(),
        };
        assert!(!p.should_fallback_from_responses_api_error(&error));
    }

    #[test]
    fn test_should_not_fallback_from_unrelated_responses_error() {
        let p = make_provider_with_responses_api("o3-mini", None);
        let error = LLMError::ResponseFormatError {
            message: "OpenAI Responses API returned error status: 500 Internal Server Error"
                .to_string(),
            raw_response: r#"{"error":{"message":"upstream overloaded"}}"#.to_string(),
        };
        assert!(!p.should_fallback_from_responses_api_error(&error));
    }

    // -- OpenAITool serialization --

    #[test]
    fn test_openai_tool_serialization() {
        let tool = OpenAITool::Function {
            tool_type: "function".to_string(),
            function: FunctionTool {
                name: "lookup".to_string(),
                description: "Lookup data".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "q": { "type": "string", "description": "query" }
                    },
                    "required": ["q"]
                }),
            },
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized.get("type"), Some(&json!("function")));
    }

    #[test]
    fn test_openai_web_search_tool_serialization() {
        let tool = OpenAITool::WebSearch {
            tool_type: "web_search".to_string(),
            user_location: Some(UserLocation {
                location_type: "approximate".to_string(),
                approximate: Some(ApproximateLocation {
                    country: "US".to_string(),
                    city: "SF".to_string(),
                    region: "CA".to_string(),
                }),
            }),
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized.get("type"), Some(&json!("web_search")));
        assert_eq!(
            serialized
                .get("user_location")
                .and_then(|v| v.get("type"))
                .and_then(|v| v.as_str()),
            Some("approximate")
        );
    }

    // -- OpenAI constructor tests --

    #[test]
    fn test_openai_new_requires_api_key() {
        let result = OpenAI::new(
            "", None, None, None, None, None, None, None, None, None, None, None, None, None, None,
            None, None, None, None, None, None,
        );
        assert!(matches!(result, Err(LLMError::AuthError(_))));
    }

    #[test]
    fn test_openai_builder_requires_api_key() {
        let result = LLMBuilder::<OpenAI>::new().build();
        assert!(result.is_err());
        if let Err(err) = result {
            assert!(err.to_string().contains("No API key provided"));
        }
    }

    #[test]
    fn test_build_function_tools_empty_returns_none() {
        let provider = make_provider("gpt-4.1");
        let tools = provider.build_function_tools(None);
        assert!(tools.is_none());
    }

    #[test]
    fn test_build_responses_function_tools_empty_returns_none() {
        let provider = make_provider("codex-mini");
        let tools = provider.build_responses_function_tools(None);
        assert!(tools.is_none());
    }

    #[test]
    fn test_responses_function_tool_serialization() {
        let tool = OpenAITool::ResponsesFunction {
            tool_type: "function".to_string(),
            name: "get_weather".to_string(),
            description: "Get weather for a city".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"}
                },
                "required": ["city"]
            }),
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["type"], "function");
        assert_eq!(serialized["name"], "get_weather");
        assert_eq!(serialized["description"], "Get weather for a city");
        assert!(serialized["parameters"].is_object());
        assert!(serialized.get("function").is_none());
    }

    #[test]
    fn test_build_responses_function_tools_produces_flat_format() {
        let provider = make_provider("codex-mini");
        let tools = vec![Tool {
            tool_type: "function".to_string(),
            function: crate::chat::FunctionTool {
                name: "search".to_string(),
                description: "Search the web".to_string(),
                parameters: json!({"type": "object", "properties": {}}),
            },
        }];
        let result = provider.build_responses_function_tools(Some(&tools)).unwrap();
        assert_eq!(result.len(), 1);
        let serialized = serde_json::to_value(&result[0]).unwrap();
        assert_eq!(serialized["type"], "function");
        assert_eq!(serialized["name"], "search");
        assert_eq!(serialized["description"], "Search the web");
        assert!(serialized.get("function").is_none());
    }

    #[test]
    fn test_build_web_search_tool_with_location() {
        let provider = OpenAI::new(
            "key",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(true),
            None,
            Some("approximate".to_string()),
            Some("US".to_string()),
            Some("SF".to_string()),
            Some("CA".to_string()),
        )
        .unwrap();

        let tool = provider.build_web_search_tool();
        match tool {
            OpenAITool::WebSearch { user_location, .. } => {
                let loc = user_location.expect("location");
                assert_eq!(loc.location_type, "approximate");
                let approx = loc.approximate.expect("approx");
                assert_eq!(approx.country, "US");
                assert_eq!(approx.city, "SF");
                assert_eq!(approx.region, "CA");
            }
            _ => panic!("expected web search tool"),
        }
    }

    // -- Chat Completions request serialization --

    #[test]
    fn test_openai_api_chat_request_serialization() {
        use crate::providers::openai_compatible::OpenAIChatMessage;
        let msg = OpenAIChatMessage {
            role: "user",
            content: Some(Right("hello".to_string())),
            tool_calls: None,
            tool_call_id: None,
        };

        let request = OpenAIAPIChatRequest {
            model: "gpt-test",
            messages: vec![msg],
            input: None,
            max_completion_tokens: Some(10),
            max_output_tokens: None,
            temperature: Some(0.2),
            stream: false,
            top_p: Some(0.9),
            top_k: Some(40),
            tools: None,
            tool_choice: Some(ToolChoice::Auto),
            reasoning_effort: None,
            response_format: None,
            stream_options: None,
            extra_body: serde_json::Map::new(),
        };

        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized.get("model"), Some(&json!("gpt-test")));
        assert_eq!(
            serialized
                .get("messages")
                .and_then(|m| m.as_array())
                .unwrap()
                .len(),
            1
        );
    }

    // -- Responses API request serialization --

    #[test]
    fn test_responses_request_serialization() {
        let request = OpenAIResponsesRequest {
            model: "codex-mini",
            input: ResponsesInput::Items(vec![InputItem::Message {
                role: "user".into(),
                content: MessageContent::Text("hello".into()),
            }]),
            instructions: Some("Be helpful"),
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            top_p: None,
            max_output_tokens: Some(1000),
            previous_response_id: None,
            reasoning: None,
            text: None,
            stream: false,
            extra_body: serde_json::Map::new(),
        };

        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized["model"], "codex-mini");
        assert_eq!(serialized["instructions"], "Be helpful");
        assert!(serialized["temperature"].as_f64().unwrap() > 0.69);
        assert_eq!(serialized["max_output_tokens"], 1000);
        assert!(!serialized["stream"].as_bool().unwrap());

        let input_items = serialized["input"].as_array().unwrap();
        assert_eq!(input_items.len(), 1);
        assert_eq!(input_items[0]["type"], "message");
        assert_eq!(input_items[0]["role"], "user");
        assert_eq!(input_items[0]["content"], "hello");
    }

    #[test]
    fn test_responses_input_string_serialization() {
        let input = ResponsesInput::Text("simple question".into());
        let serialized = serde_json::to_value(&input).unwrap();
        assert_eq!(serialized, json!("simple question"));
    }

    #[test]
    fn test_input_item_function_call_serialization() {
        let item = InputItem::FunctionCall {
            call_id: "call_123".into(),
            name: "get_weather".into(),
            arguments: r#"{"city":"SF"}"#.into(),
        };
        let serialized = serde_json::to_value(&item).unwrap();
        assert_eq!(serialized["type"], "function_call");
        assert_eq!(serialized["call_id"], "call_123");
        assert_eq!(serialized["name"], "get_weather");
    }

    #[test]
    fn test_input_item_function_call_output_serialization() {
        let item = InputItem::FunctionCallOutput {
            call_id: "call_123".into(),
            output: "72F and sunny".into(),
        };
        let serialized = serde_json::to_value(&item).unwrap();
        assert_eq!(serialized["type"], "function_call_output");
        assert_eq!(serialized["call_id"], "call_123");
        assert_eq!(serialized["output"], "72F and sunny");
    }

    #[test]
    fn test_content_part_input_text_serialization() {
        let part = ContentPart::InputText {
            text: "hello world".into(),
        };
        let serialized = serde_json::to_value(&part).unwrap();
        assert_eq!(serialized["type"], "input_text");
        assert_eq!(serialized["text"], "hello world");
    }

    #[test]
    fn test_content_part_input_image_url_serialization() {
        let part = ContentPart::InputImage {
            source: ImageSource::Url {
                url: "https://example.com/image.png".into(),
            },
        };
        let serialized = serde_json::to_value(&part).unwrap();
        assert_eq!(serialized["type"], "input_image");
        assert_eq!(serialized["url"], "https://example.com/image.png");
    }

    // -- Responses API response deserialization --

    #[test]
    fn test_responses_response_text_deserialization() {
        let json = json!({
            "id": "resp_123",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!", "annotations": []}]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });

        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.id, "resp_123");
        assert_eq!(resp.status, "completed");
        assert_eq!(resp.text(), Some("Hello!".into()));
        assert!(resp.tool_calls().is_none());
        let usage = resp.usage().unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
    }

    #[test]
    fn test_responses_response_tool_calls_deserialization() {
        let json = json!({
            "id": "resp_456",
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": "{\"city\":\"SF\"}"
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
        });

        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert!(resp.text().is_none());
        let calls = resp.tool_calls().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_abc");
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn test_responses_response_unknown_output_type() {
        let json = json!({
            "id": "resp_789",
            "status": "completed",
            "output": [
                {"type": "some_future_type", "id": "ft_1", "data": "whatever"},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });

        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.output.len(), 2);
        assert!(matches!(resp.output[0], OutputItem::Unknown));
        assert_eq!(resp.text(), Some("Hi".into()));
    }

    #[test]
    fn test_responses_response_refusal() {
        let json = json!({
            "id": "resp_ref",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "I can't do that"}]
            }],
            "usage": {"input_tokens": 5, "output_tokens": 5, "total_tokens": 10}
        });

        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert!(resp.text().is_none()); // refusals don't appear as text
    }

    // -- Responses text format --

    #[test]
    fn test_responses_text_format_from_structured_output() {
        let sof = StructuredOutputFormat {
            name: "MySchema".into(),
            description: Some("A schema".into()),
            schema: Some(json!({"type": "object", "properties": {"x": {"type": "string"}}})),
            strict: Some(true),
        };
        let format: ResponsesTextFormat = sof.into();
        let serialized = serde_json::to_value(&format).unwrap();
        assert_eq!(serialized["format"]["type"], "json_schema");
        assert_eq!(serialized["format"]["name"], "MySchema");
        assert_eq!(serialized["format"]["strict"], true);
        assert!(
            serialized["format"]["schema"]["additionalProperties"]
                .as_bool()
                .is_some()
        );
    }

    // -- prepare_responses_input --

    #[test]
    fn test_prepare_responses_input_system_becomes_instructions() {
        let p = make_provider("codex-mini");
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                message_type: MessageType::Text,
                content: "You are helpful".into(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: "Hi".into(),
            },
        ];

        let (instructions, items) = p.prepare_responses_input(&messages);
        assert_eq!(instructions, Some("You are helpful".into()));
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Message { role, content } => {
                assert_eq!(role, "user");
                match content {
                    MessageContent::Text(t) => assert_eq!(t, "Hi"),
                    _ => panic!("expected text"),
                }
            }
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn test_prepare_responses_input_tool_use_and_result() {
        let p = make_provider("codex-mini");
        let tc = ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"SF"}"#.into(),
            },
        };

        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(vec![tc.clone()]),
                content: String::new(),
            },
            ChatMessage {
                role: ChatRole::Tool,
                message_type: MessageType::ToolResult(vec![tc]),
                content: "72F".into(),
            },
        ];

        let (instructions, items) = p.prepare_responses_input(&messages);
        assert!(instructions.is_none());
        assert_eq!(items.len(), 2);

        match &items[0] {
            InputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"SF"}"#);
            }
            _ => panic!("expected function_call"),
        }

        match &items[1] {
            InputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(output, "72F");
            }
            _ => panic!("expected function_call_output"),
        }
    }

    #[test]
    fn test_prepare_responses_input_tool_result_falls_back_to_tool_call_arguments() {
        let p = make_provider("codex-mini");
        let tc = ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::FunctionCall {
                name: "get_weather".into(),
                arguments: "72F".into(),
            },
        };

        let messages = vec![ChatMessage {
            role: ChatRole::Tool,
            message_type: MessageType::ToolResult(vec![tc]),
            content: String::new(),
        }];

        let (_, items) = p.prepare_responses_input(&messages);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::FunctionCallOutput { call_id, output } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(output, "72F");
            }
            _ => panic!("expected function_call_output"),
        }
    }

    #[test]
    fn test_prepare_responses_input_image_url() {
        let p = make_provider("codex-mini");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("https://example.com/img.png".into()),
            content: String::new(),
        }];

        let (_, items) = p.prepare_responses_input(&messages);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Message { content, .. } => match content {
                MessageContent::Parts(parts) => {
                    assert_eq!(parts.len(), 1);
                    match &parts[0] {
                        ContentPart::InputImage {
                            source: ImageSource::Url { url },
                        } => assert_eq!(url, "https://example.com/img.png"),
                        _ => panic!("expected url image"),
                    }
                }
                _ => panic!("expected parts"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn test_prepare_responses_input_image_url_with_text() {
        let p = make_provider("codex-mini");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::ImageURL("https://example.com/img.png".into()),
            content: "Describe this image".into(),
        }];

        let (_, items) = p.prepare_responses_input(&messages);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Message { content, .. } => match content {
                MessageContent::Parts(parts) => {
                    assert_eq!(parts.len(), 2);
                    match &parts[0] {
                        ContentPart::InputText { text } => assert_eq!(text, "Describe this image"),
                        _ => panic!("expected input_text first"),
                    }
                    match &parts[1] {
                        ContentPart::InputImage {
                            source: ImageSource::Url { url },
                        } => assert_eq!(url, "https://example.com/img.png"),
                        _ => panic!("expected url image second"),
                    }
                }
                _ => panic!("expected parts"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn test_prepare_responses_input_image_base64() {
        let p = make_provider("codex-mini");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Image((ImageMime::PNG, vec![1, 2, 3])),
            content: String::new(),
        }];

        let (_, items) = p.prepare_responses_input(&messages);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Message { content, .. } => match content {
                MessageContent::Parts(parts) => {
                    assert_eq!(parts.len(), 1);
                    match &parts[0] {
                        ContentPart::InputImage {
                            source: ImageSource::Base64 { media_type, .. },
                        } => assert_eq!(media_type, "image/png"),
                        _ => panic!("expected base64 image"),
                    }
                }
                _ => panic!("expected parts"),
            },
            _ => panic!("expected message"),
        }
    }

    #[test]
    fn test_prepare_responses_input_image_base64_with_text() {
        let p = make_provider("codex-mini");
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Image((ImageMime::PNG, vec![1, 2, 3])),
            content: "Analyze this image".into(),
        }];

        let (_, items) = p.prepare_responses_input(&messages);
        assert_eq!(items.len(), 1);
        match &items[0] {
            InputItem::Message { content, .. } => match content {
                MessageContent::Parts(parts) => {
                    assert_eq!(parts.len(), 2);
                    match &parts[0] {
                        ContentPart::InputText { text } => assert_eq!(text, "Analyze this image"),
                        _ => panic!("expected input_text first"),
                    }
                    match &parts[1] {
                        ContentPart::InputImage {
                            source: ImageSource::Base64 { media_type, .. },
                        } => assert_eq!(media_type, "image/png"),
                        _ => panic!("expected base64 image second"),
                    }
                }
                _ => panic!("expected parts"),
            },
            _ => panic!("expected message"),
        }
    }

    // -- Streaming parser tests --

    #[test]
    fn test_stream_parser_text_delta() {
        let mut parser = ResponsesStreamParser::new();

        // Add a message item
        let added = json!({
            "output_index": 0,
            "item": {"id": "msg_1", "type": "message", "role": "assistant"}
        });
        let chunks = parser.process_event("response.output_item.added", &added.to_string());
        assert!(chunks.is_empty()); // message items don't emit start events

        // Text delta
        let delta = json!({"item_id": "msg_1", "content_index": 0, "delta": "Hello"});
        let chunks = parser.process_event("response.output_text.delta", &delta.to_string());
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            Ok(StreamChunk::Text(t)) => assert_eq!(t, "Hello"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    #[test]
    fn test_stream_parser_function_call() {
        let mut parser = ResponsesStreamParser::new();

        // Add a function_call item
        let added = json!({
            "output_index": 0,
            "item": {
                "id": "fc_1",
                "type": "function_call",
                "name": "get_weather",
                "call_id": "call_abc"
            }
        });
        let chunks = parser.process_event("response.output_item.added", &added.to_string());
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            Ok(StreamChunk::ToolUseStart { index, id, name }) => {
                assert_eq!(*index, 0);
                assert_eq!(id, "call_abc");
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolUseStart, got {other:?}"),
        }

        // Argument deltas
        let delta1 = json!({"item_id": "fc_1", "delta": r#"{"ci"#});
        let chunks = parser.process_event(
            "response.function_call_arguments.delta",
            &delta1.to_string(),
        );
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            Ok(StreamChunk::ToolUseInputDelta {
                index,
                partial_json,
            }) => {
                assert_eq!(*index, 0);
                assert_eq!(partial_json, r#"{"ci"#);
            }
            other => panic!("expected ToolUseInputDelta, got {other:?}"),
        }

        // Arguments done
        let done = json!({"item_id": "fc_1", "arguments": r#"{"city":"SF"}"#});
        let chunks =
            parser.process_event("response.function_call_arguments.done", &done.to_string());
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            Ok(StreamChunk::ToolUseComplete { index, tool_call }) => {
                assert_eq!(*index, 0);
                assert_eq!(tool_call.id, "call_abc");
                assert_eq!(tool_call.function.name, "get_weather");
                assert_eq!(tool_call.function.arguments, r#"{"city":"SF"}"#);
            }
            other => panic!("expected ToolUseComplete, got {other:?}"),
        }
    }

    #[test]
    fn test_stream_parser_completed_with_usage() {
        let mut parser = ResponsesStreamParser::new();

        let completed = json!({
            "response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            }
        });
        let chunks = parser.process_event("response.completed", &completed.to_string());
        assert_eq!(chunks.len(), 2);
        match &chunks[0] {
            Ok(StreamChunk::Usage(u)) => {
                assert_eq!(u.prompt_tokens, 10);
                assert_eq!(u.completion_tokens, 5);
            }
            other => panic!("expected Usage, got {other:?}"),
        }
        match &chunks[1] {
            Ok(StreamChunk::Done { stop_reason }) => assert_eq!(stop_reason, "end_turn"),
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn test_stream_parser_completed_with_tool_use() {
        let mut parser = ResponsesStreamParser::new();

        // First add a function call to set has_function_calls
        let added = json!({
            "output_index": 0,
            "item": {"id": "fc_1", "type": "function_call", "name": "fn", "call_id": "c1"}
        });
        parser.process_event("response.output_item.added", &added.to_string());

        let completed = json!({
            "response": {
                "id": "resp_1",
                "status": "completed",
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            }
        });
        let chunks = parser.process_event("response.completed", &completed.to_string());
        let done = chunks
            .iter()
            .find(|c| matches!(c, Ok(StreamChunk::Done { .. })));
        match done {
            Some(Ok(StreamChunk::Done { stop_reason })) => assert_eq!(stop_reason, "tool_use"),
            other => panic!("expected Done with tool_use, got {other:?}"),
        }
    }

    #[test]
    fn test_stream_parser_failed() {
        let mut parser = ResponsesStreamParser::new();
        let failed = json!({
            "response": {"error": {"message": "Rate limit exceeded"}}
        });
        let chunks = parser.process_event("response.failed", &failed.to_string());
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_err());
    }

    #[test]
    fn test_stream_parser_incomplete() {
        let mut parser = ResponsesStreamParser::new();
        let chunks = parser.process_event("response.incomplete", "{}");
        assert_eq!(chunks.len(), 1);
        match &chunks[0] {
            Ok(StreamChunk::Done { stop_reason }) => assert_eq!(stop_reason, "max_tokens"),
            other => panic!("expected Done max_tokens, got {other:?}"),
        }
    }

    #[test]
    fn test_stream_parser_unknown_event_ignored() {
        let mut parser = ResponsesStreamParser::new();
        let chunks = parser.process_event("response.some_future_event", r#"{"foo":"bar"}"#);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_responses_chunk_to_stream_response_maps_tool_use_complete() {
        let tool_call = ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::FunctionCall {
                name: "lookup".into(),
                arguments: r#"{"q":"x"}"#.into(),
            },
        };
        let mapped = OpenAI::responses_chunk_to_stream_response(Ok(StreamChunk::ToolUseComplete {
            index: 0,
            tool_call: tool_call.clone(),
        }))
        .expect("expected mapped value")
        .expect("expected Ok value");

        assert_eq!(mapped.choices.len(), 1);
        assert!(mapped.choices[0].delta.content.is_none());
        let calls = mapped.choices[0]
            .delta
            .tool_calls
            .as_ref()
            .expect("expected tool calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "call_1");
        assert_eq!(calls[0].function.name, "lookup");
        assert_eq!(calls[0].function.arguments, r#"{"q":"x"}"#);
    }

    // -- Builder test --

    #[test]
    fn test_builder_use_responses_api() {
        let builder = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("gpt-4.1")
            .use_responses_api(true);
        let openai = builder.build().unwrap();
        assert!(openai.should_use_responses_api());
    }

    #[test]
    fn test_builder_default_auto_detect() {
        let openai = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("codex-mini")
            .build()
            .unwrap();
        assert!(openai.should_use_responses_api());

        let openai = LLMBuilder::<OpenAI>::new()
            .api_key("test-key")
            .model("gpt-4.1")
            .build()
            .unwrap();
        assert!(!openai.should_use_responses_api());
    }

    // -- Phase 3: Built-in tool types --

    #[test]
    fn test_file_search_tool_serialization() {
        let tool = OpenAITool::FileSearch {
            tool_type: "file_search".to_string(),
            vector_store_ids: vec!["vs_abc".to_string()],
            max_num_results: Some(5),
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["type"], "file_search");
        assert_eq!(serialized["vector_store_ids"][0], "vs_abc");
        assert_eq!(serialized["max_num_results"], 5);
    }

    #[test]
    fn test_code_interpreter_tool_serialization() {
        let tool = OpenAITool::CodeInterpreter {
            tool_type: "code_interpreter".to_string(),
        };
        let serialized = serde_json::to_value(&tool).unwrap();
        assert_eq!(serialized["type"], "code_interpreter");
    }

    #[test]
    fn test_annotation_deserialization() {
        let json = json!({
            "type": "url_citation",
            "start_index": 10,
            "end_index": 50,
            "url": "https://example.com",
            "title": "Example Page"
        });
        let ann: Annotation = serde_json::from_value(json).unwrap();
        assert_eq!(ann.annotation_type, Some("url_citation".into()));
        assert_eq!(ann.start_index, Some(10));
        assert_eq!(ann.end_index, Some(50));
        assert_eq!(ann.url, Some("https://example.com".into()));
        assert_eq!(ann.title, Some("Example Page".into()));
        assert!(ann.file_id.is_none());
    }

    #[test]
    fn test_output_text_with_annotations() {
        let json = json!({
            "id": "resp_ann",
            "status": "completed",
            "output": [{
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": "Here is info [1].",
                    "annotations": [{
                        "type": "url_citation",
                        "start_index": 13,
                        "end_index": 16,
                        "url": "https://example.com",
                        "title": "Source"
                    }]
                }]
            }],
            "usage": {"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
        });
        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.text(), Some("Here is info [1].".into()));
        if let OutputItem::Message { content, .. } = &resp.output[0] {
            if let OutputContentPart::Text { annotations, .. } = &content[0] {
                assert_eq!(annotations.len(), 1);
                assert_eq!(annotations[0].url, Some("https://example.com".into()));
            } else {
                panic!("expected text content");
            }
        } else {
            panic!("expected message");
        }
    }

    #[test]
    fn test_web_search_call_output_item() {
        let json = json!({
            "id": "resp_ws",
            "status": "completed",
            "output": [
                {"type": "web_search_call", "id": "ws_1", "status": "completed"},
                {"type": "message", "id": "msg_1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "Result", "annotations": []}]}
            ],
            "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}
        });
        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.output.len(), 2);
        match &resp.output[0] {
            OutputItem::WebSearchCall { id, status } => {
                assert_eq!(id, "ws_1");
                assert_eq!(status, "completed");
            }
            _ => panic!("expected WebSearchCall"),
        }
        assert_eq!(resp.text(), Some("Result".into()));
    }

    // -- Phase 4: response_id, thinking, previous_response_id --

    #[test]
    fn test_response_id() {
        let json = json!({
            "id": "resp_abc123",
            "status": "completed",
            "output": [{"type": "message", "id": "m1", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert_eq!(resp.response_id(), Some("resp_abc123"));
    }

    #[test]
    fn test_thinking_from_reasoning_summary() {
        let json = json!({
            "id": "resp_think",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [
                        {"type": "summary_text", "text": "The user asked about weather."},
                        {"type": "summary_text", "text": "I should check the forecast."}
                    ]
                },
                {"type": "message", "id": "m1", "role": "assistant",
                 "content": [{"type": "output_text", "text": "It's sunny!", "annotations": []}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        });
        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        let thinking = resp.thinking().unwrap();
        assert!(thinking.contains("The user asked about weather."));
        assert!(thinking.contains("I should check the forecast."));
        assert_eq!(resp.text(), Some("It's sunny!".into()));
    }

    #[test]
    fn test_thinking_none_when_no_reasoning() {
        let json = json!({
            "id": "resp_no_think",
            "status": "completed",
            "output": [{"type": "message", "id": "m1", "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hi", "annotations": []}]}],
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7}
        });
        let resp: OpenAIResponsesResponse = serde_json::from_value(json).unwrap();
        assert!(resp.thinking().is_none());
    }

    #[test]
    fn test_previous_response_id_in_request() {
        let request = OpenAIResponsesRequest {
            model: "codex-mini",
            input: ResponsesInput::Items(vec![InputItem::Message {
                role: "user".into(),
                content: MessageContent::Text("follow up".into()),
            }]),
            instructions: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            previous_response_id: Some("resp_prev_123"),
            reasoning: None,
            text: None,
            stream: false,
            extra_body: serde_json::Map::new(),
        };
        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized["previous_response_id"], "resp_prev_123");
    }

    #[test]
    fn test_previous_response_id_omitted_when_none() {
        let request = OpenAIResponsesRequest {
            model: "codex-mini",
            input: ResponsesInput::Text("hello".into()),
            instructions: None,
            tools: None,
            tool_choice: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            previous_response_id: None,
            reasoning: None,
            text: None,
            stream: false,
            extra_body: serde_json::Map::new(),
        };
        let serialized = serde_json::to_value(&request).unwrap();
        assert!(serialized.get("previous_response_id").is_none());
    }

    #[test]
    fn test_previous_response_id_field_on_openai() {
        let mut p = make_provider("codex-mini");
        assert!(p.previous_response_id.is_none());
        p.previous_response_id = Some("resp_abc".into());
        assert_eq!(p.previous_response_id, Some("resp_abc".into()));
    }

    #[test]
    fn test_reasoning_config_serialization() {
        let config = ResponsesReasoningConfig {
            effort: Some("high".into()),
            summary: Some("auto".into()),
        };
        let serialized = serde_json::to_value(&config).unwrap();
        assert_eq!(serialized["effort"], "high");
        assert_eq!(serialized["summary"], "auto");
    }

    #[test]
    fn test_reasoning_summary_part_deserialization() {
        let json = json!({"type": "summary_text", "text": "Some reasoning"});
        let part: ReasoningSummaryPart = serde_json::from_value(json).unwrap();
        assert_eq!(part.part_type, Some("summary_text".into()));
        assert_eq!(part.text, Some("Some reasoning".into()));
    }
}
