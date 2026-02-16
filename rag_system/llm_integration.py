"""LLM integration module."""

from __future__ import annotations

import os
from typing import List, Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import BitsAndBytesConfig


def _load_hf_token_from_env() -> Optional[str]:
    """Load HuggingFace token from .env file (optional - only needed for gated models)."""
    load_dotenv(os.path.join("./.env.txt"))
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("ℹ️  No HuggingFace token found (optional - only needed for gated models like Phi-3)")
    return hf_token
  

class RAGPrompt:
    """Manages RAG prompts and response generation."""
    
    @staticmethod
    def get_system_prompt() -> str:
        """Get the system prompt for the RAG system."""
        return (
            "You are an assistant that answers user questions using only the provided context. "
            "Cite sources and avoid hallucination. If the answer is not in the context, respond "
            "that the information is not available in the provided documents."
        )
    
    @staticmethod
    def create_answer_prompt(query: str, context: str) -> str:
        """
        Create a prompt for answering a question with context.
        
        Args:
            query: The user's question
            context: Retrieved context from documents
            
        Returns:
            Formatted prompt string
        """
        return (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer concisely and include citations (e.g., [DocName - Page X])."
        )
    
    @staticmethod
    def format_context(retrieved_chunks: List[Dict]) -> str:
        """
        Convert list of retrieved chunk dicts into a single context string with simple citations.
        Expects each chunk to have at least 'text' and optional 'document'/'page' metadata.
        """
        parts = []
        for i, c in enumerate(retrieved_chunks, start=1):
            text = c.get("text", c.get("content", "")).strip()
            doc = c.get("document") or c.get("source") or c.get("doc_name") or "UnknownDoc"
            page = c.get("page")
            citation = f"[{doc}" + (f" - Page {page}]" if page is not None else "]")
            parts.append(f"{citation}\n{text}")
        return "\n\n---\n\n".join(parts)


class HFBackend:
    """Hugging Face backend for text generation."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.3,
        max_new_tokens: int = 200,
        quantize_4bit: bool = True,
        local_model_path: str = None,  # NEW: Allow loading from local path
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Determine if loading from local path (Kaggle dataset) or HuggingFace
        if local_model_path and os.path.exists(local_model_path):
            print(f"📂 Loading model from local path: {local_model_path}")
            model_path = local_model_path
            use_auth_token = None
            local_files_only = True
        else:
            print(f"🌐 Loading model from HuggingFace: {model_name}")
            print(f"   (Will use cached version if available)")
            use_auth_token = _load_hf_token_from_env()
            model_path = model_name
            local_files_only = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=use_auth_token,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        # Load model config
        print(f"📝 Loading model config...")
        config = AutoConfig.from_pretrained(
            model_path,
            token=use_auth_token,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )

        load_kwargs: Dict[str, object] = {
            "config": config,
            "device_map": "auto",
            "trust_remote_code": True,
            "local_files_only": local_files_only,
        }
        
        if use_auth_token:
            load_kwargs["token"] = use_auth_token

        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        if quantize_4bit and torch.cuda.is_available():
            if _has_bitsandbytes():
                print("⚡ Loading model in 4-bit quantized mode")

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

                load_kwargs["quantization_config"] = bnb_config

                # IMPORTANT: remove torch_dtype when using 4bit
                load_kwargs.pop("torch_dtype", None)

            else:
                print("⚠ bitsandbytes not available, loading full precision model.")


        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **load_kwargs,
        )

        self.model.eval()

    def generate(self, user_prompt: str, system_prompt: str) -> str:
        prompt_text = self._format_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        device = _get_model_device(self.model)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        do_sample = self.temperature is not None and self.temperature > 0

        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=do_sample,
            temperature=self.temperature if do_sample else None,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract only the generated answer (remove prompt)
        # For chat templates (Mistral uses [INST]...[/INST] format)
        if "[/INST]" in output_text:
            output_text = output_text.split("[/INST]")[-1].strip()
        elif output_text.startswith(prompt_text):
            # Fallback for non-chat-template models
            output_text = output_text[len(prompt_text):]
        
        return output_text.strip()

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return f"{system_prompt}\n\n{user_prompt}\n\nAnswer:"


class LLMIntegration:
    """Integrates LLM with RAG pipeline."""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.3,
        max_new_tokens: int = 300,
        backend: str = "hf",
        local_model_path: str = None,  # Path to local model (Kaggle dataset or local cache)
    ):
        """
        Initialize LLM integration.
        
        Args:
            model_name: Name of the Hugging Face model to use
            temperature: Temperature for generation (0-1)
            max_new_tokens: Maximum tokens to generate
            backend: Backend name (only "hf" supported right now)
            local_model_path: Path to local model directory (optional, for Kaggle datasets)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.backend = backend
        
        if backend != "hf":
            raise ValueError(f"Unsupported backend: {backend}. Use backend=\"hf\".")

        self.llm = HFBackend(
            model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            quantize_4bit=True,
            local_model_path=local_model_path,  # Pass through to backend
        )
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        prompt = RAGPrompt.create_answer_prompt(query, context)
        system_prompt = RAGPrompt.get_system_prompt()

        try:
            response = self.llm.generate(prompt, system_prompt)
            return response.strip()
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Unable to generate answer at this time."


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _get_model_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
