// This example demonstrates OpenAI's Responses API features:
// - Auto-detection (codex models use Responses API automatically)
// - Explicit opt-in via use_responses_api(true)
// - Conversation chaining via previous_response_id
// - Streaming with the Responses API
use autoagents::llm::backends::openai::OpenAI;
use autoagents::llm::builder::LLMBuilder;
use autoagents::llm::chat::{ChatMessage, ChatProvider};
use autoagents::prelude::Error;
use std::sync::Arc;
use tokio_stream::StreamExt;

pub async fn run() -> Result<(), Error> {
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "".into());

    // Build an OpenAI client with Responses API explicitly enabled.
    // For codex models (codex-mini, etc.) this is auto-detected.
    let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
        .api_key(&api_key)
        .model("gpt-4.1-mini")
        .max_tokens(256)
        .use_responses_api(true)
        .build()
        .expect("Failed to build LLM");

    // --- Non-streaming chat ---
    println!("--- Non-streaming Responses API ---");
    let message = ChatMessage::user()
        .content("What is the capital of France? Reply in one sentence.")
        .build();
    let response = llm.chat(std::slice::from_ref(&message), None).await?;
    println!("Response: {}", response);

    // The response includes a response_id that can be used for conversation chaining
    if let Some(id) = response.response_id() {
        println!("Response ID: {id}");
    }

    // --- Streaming chat ---
    println!("\n--- Streaming Responses API ---");
    let message = ChatMessage::user()
        .content("Name three famous landmarks in Paris.")
        .build();
    let mut stream = llm
        .chat_stream(std::slice::from_ref(&message), None)
        .await?;
    print!("Streaming: ");
    while let Some(result) = stream.next().await {
        match result {
            Ok(text) => print!("{text}"),
            Err(e) => eprintln!("\nStream error: {e}"),
        }
    }
    println!();

    // --- Conversation chaining ---
    println!("\n--- Conversation chaining via previous_response_id ---");

    // For chaining, we need a mutable reference to set previous_response_id.
    // Use Arc::get_mut or build without Arc for multi-turn conversations.
    let mut llm = OpenAI::new(
        &api_key,
        None,
        Some("gpt-4.1-mini".into()),
        Some(256),
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
    .expect("Failed to create OpenAI client");
    llm.use_responses_api = Some(true);

    // First message
    let msg1 = ChatMessage::user()
        .content("My favorite color is blue. Remember that.")
        .build();
    let response1 = llm.chat(std::slice::from_ref(&msg1), None).await?;
    println!("Turn 1: {response1}");

    // Chain the conversation using the response ID
    llm.chain_response(&*response1);
    println!(
        "Chained with response_id: {}",
        llm.previous_response_id.as_deref().unwrap_or("none")
    );

    // Follow-up referencing the previous context
    let msg2 = ChatMessage::user()
        .content("What is my favorite color?")
        .build();
    let response2 = llm.chat(std::slice::from_ref(&msg2), None).await?;
    println!("Turn 2: {response2}");

    println!("\nResponses API example completed!");
    Ok(())
}
