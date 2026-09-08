import os
import json
from typing import List, Dict, Any
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Keeping your existing embedding setup
embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2", 
    api_key=os.getenv("GOOGLE_API_KEY")
)

def partition_document(file_path: str):
    print(f"Partitioning document: {file_path}")
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res", 
        infer_table_structure=True, 
        extract_image_block_types=["Image"], 
        extract_image_block_to_payload=True 
    )
    return elements

def separate_content_types(chunk) -> Dict[str, Any]:
    content_data = {
        'text': chunk.text,
        'tables': [],
        'images': [],
        'types': ['text']
    }
    
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            element_type = type(element).__name__
            
            if element_type == 'Table':
                content_data['types'].append('table')
                table_html = getattr(element.metadata, 'text_as_html', element.text)
                content_data['tables'].append(table_html)
            
            elif element_type == 'Image':
                if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                    content_data['types'].append('image')
                    content_data['images'].append(element.metadata.image_base64)
    
    content_data['types'] = list(set(content_data['types']))
    return content_data

def create_ai_enhanced_summary(text: str, tables: List[str], images: List[str]) -> str:
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        prompt_text = f"You are creating a searchable description for document content retrieval.\n\nTEXT CONTENT:\n{text}\n\n"
        
        if tables:
            prompt_text += "TABLES:\n"
            for i, table in enumerate(tables):
                prompt_text += f"Table {i+1}:\n{table}\n\n"
        
        prompt_text += "Generate a comprehensive, searchable description covering key facts, data, topics, and visual analysis."

        # Format prompt for Groq Vision Model (Llama 3.2 Vision)
        messages_content = [{"type": "text", "text": prompt_text}]
        for image_base64 in images:
            messages_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.2,
            max_tokens=1024,
        )
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"AI summary failed: {e}")
        summary = f"{text[:300]}..."
        if tables: summary += f" [Contains {len(tables)} table(s)]"
        if images: summary += f" [Contains {len(images)} image(s)]"
        return summary

def load_and_chunk_pdf(path: str) -> List[Dict[str, Any]]:
    elements = partition_document(path)
    chunks = chunk_by_title(
        elements,
        max_characters=3000,
        new_after_n_chars=2400,
        combine_text_under_n_chars=500
    )
    
    processed_chunks = []
    for chunk in chunks:
        content_data = separate_content_types(chunk)
        
        if content_data['tables'] or content_data['images']:
            enhanced_content = create_ai_enhanced_summary(
                content_data['text'],
                content_data['tables'], 
                content_data['images']
            )
        else:
            enhanced_content = content_data['text']
            
        processed_chunks.append({
            "enhanced_content": enhanced_content,
            "original_content": {
                "raw_text": content_data['text'],
                "tables_html": content_data['tables'],
                "images_base64": content_data['images']
            }
        })
        
    return processed_chunks

def embed_texts(texts: List[str]) -> List[List[float]]:
    return embed_model.get_text_embedding_batch(texts)
