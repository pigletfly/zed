mod models;

use rand::Rng;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;

use anyhow::{Context, Error, Result, anyhow};
use aws_sdk_bedrockruntime as bedrock;
pub use aws_sdk_bedrockruntime as bedrock_client;
pub use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockInnerContent, SpecificToolChoice as BedrockSpecificTool,
    ToolChoice as BedrockToolChoice, ToolInputSchema as BedrockToolInputSchema,
    ToolSpecification as BedrockTool,
};
use aws_smithy_types::{Document, Number as AwsNumber};
pub use bedrock::operation::converse_stream::ConverseStreamInput as BedrockStreamingRequest;
pub use bedrock::types::{
    ContentBlock as BedrockRequestContent, ConversationRole as BedrockRole,
    ConverseOutput as BedrockResponse, ConverseStreamOutput as BedrockStreamingResponse,
    Message as BedrockMessage, ResponseStream as BedrockResponseStream,
};
use futures::stream::{self, BoxStream, Stream};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};
use thiserror::Error;

pub use crate::models::*;

pub async fn complete(
    client: &bedrock::Client,
    request: Request,
) -> Result<BedrockResponse, BedrockError> {
    let response = bedrock::Client::converse(client)
        .model_id(request.model.clone())
        .set_messages(request.messages.into())
        .send()
        .await
        .context("failed to send request to Bedrock");

    match response {
        Ok(output) => output
            .output
            .ok_or_else(|| BedrockError::Other(anyhow!("no output"))),
        Err(err) => Err(BedrockError::Other(err)),
    }
}

async fn exponential_backoff_with_jitter(
    attempt: u32,
    base_delay: Duration,
    max_delay: Duration,
) -> Duration {
    let base = (2_u32).pow(attempt) as f64;
    let max_delay_ms = max_delay.as_millis() as f64;
    let delay_ms = (base * base_delay.as_millis() as f64).min(max_delay_ms);

    // Add random jitter between 0% and 100% of the delay
    let jitter = rand::thread_rng().gen_range(0.0..1.0);
    let final_delay_ms = (delay_ms * jitter) as u64;

    Duration::from_millis(final_delay_ms)
}

pub async fn stream_completion(
    client: bedrock::Client,
    request: Request,
    handle: tokio::runtime::Handle,
) -> Result<BoxStream<'static, Result<BedrockStreamingResponse, BedrockError>>, Error> {
    const MAX_RETRIES: u32 = 3;
    const BASE_DELAY: Duration = Duration::from_millis(100);
    const MAX_DELAY: Duration = Duration::from_secs(10);

    let mut attempt = 0;
    let client = Arc::new(client);
    let model = Arc::new(request.model);
    let messages = Arc::new(request.messages);

    loop {
        let client = client.clone();
        let model = model.clone();
        let messages = messages.clone();

        match handle
            .spawn(async move {
                let response = bedrock::Client::converse_stream(&client)
                    .model_id((*model).clone())
                    .set_messages((*messages).clone().into())
                    .send()
                    .await;

                match response {
                    Ok(output) => {
                        let stream: Pin<
                            Box<
                                dyn Stream<Item = Result<BedrockStreamingResponse, BedrockError>>
                                    + Send,
                            >,
                        > = Box::pin(stream::unfold(output.stream, |mut stream| async move {
                            match stream.recv().await {
                                Ok(Some(output)) => Some((Ok(output), stream)),
                                Ok(None) => None,
                                Err(err) => {
                                    let error =
                                        aws_sdk_bedrockruntime::error::DisplayErrorContext(err);
                                    let error_str = format!("{:?}", error);
                                    let error = if error_str.contains("ThrottlingException") {
                                        BedrockError::ThrottlingError
                                    } else {
                                        BedrockError::ClientError(anyhow!("{:?}", error))
                                    };
                                    Some((Err(error), stream))
                                }
                            }
                        }));
                        Ok(stream)
                    }
                    Err(err) => {
                        let error = aws_sdk_bedrockruntime::error::DisplayErrorContext(err);
                        let error_str = format!("{:?}", error);
                        if error_str.contains("ThrottlingException") {
                            Err(BedrockError::ThrottlingError.into())
                        } else {
                            Err(anyhow!("{:?}", error))
                        }
                    }
                }
            })
            .await
        {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(err)) => {
                if let Some(bedrock_err) = err.downcast_ref::<BedrockError>() {
                    match bedrock_err {
                        BedrockError::ThrottlingError if attempt < MAX_RETRIES => {
                            let delay =
                                exponential_backoff_with_jitter(attempt, BASE_DELAY, MAX_DELAY)
                                    .await;
                            sleep(delay).await;
                            attempt += 1;
                            continue;
                        }
                        _ => return Err(err),
                    }
                } else {
                    return Err(err);
                }
            }
            Err(err) => return Err(anyhow!("failed to spawn task: {err:?}")),
        }
    }
}

pub fn aws_document_to_value(document: &Document) -> Value {
    match document {
        Document::Null => Value::Null,
        Document::Bool(value) => Value::Bool(*value),
        Document::Number(value) => match *value {
            AwsNumber::PosInt(value) => Value::Number(Number::from(value)),
            AwsNumber::NegInt(value) => Value::Number(Number::from(value)),
            AwsNumber::Float(value) => Value::Number(Number::from_f64(value).unwrap()),
        },
        Document::String(value) => Value::String(value.clone()),
        Document::Array(array) => Value::Array(array.iter().map(aws_document_to_value).collect()),
        Document::Object(map) => Value::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), aws_document_to_value(value)))
                .collect(),
        ),
    }
}

pub fn value_to_aws_document(value: &Value) -> Document {
    match value {
        Value::Null => Document::Null,
        Value::Bool(value) => Document::Bool(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_u64() {
                Document::Number(AwsNumber::PosInt(value))
            } else if let Some(value) = value.as_i64() {
                Document::Number(AwsNumber::NegInt(value))
            } else if let Some(value) = value.as_f64() {
                Document::Number(AwsNumber::Float(value))
            } else {
                Document::Null
            }
        }
        Value::String(value) => Document::String(value.clone()),
        Value::Array(array) => Document::Array(array.iter().map(value_to_aws_document).collect()),
        Value::Object(map) => Document::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), value_to_aws_document(value)))
                .collect(),
        ),
    }
}

#[derive(Debug)]
pub struct Request {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<BedrockMessage>,
    pub tools: Vec<BedrockTool>,
    pub tool_choice: Option<BedrockToolChoice>,
    pub system: Option<String>,
    pub metadata: Option<Metadata>,
    pub stop_sequences: Vec<String>,
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    pub user_id: Option<String>,
}

#[derive(Error, Debug)]
pub enum BedrockError {
    #[error("client error: {0}")]
    ClientError(anyhow::Error),
    #[error("extension error: {0}")]
    ExtensionError(anyhow::Error),
    #[error("throttling error: request was denied due to exceeding the account quotas")]
    ThrottlingError,
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
