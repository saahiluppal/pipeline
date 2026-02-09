import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import asyncio
import requests
from fastapi import FastAPI, File, UploadFile, Form
from loguru import logger

import zipfile
import tempfile
from typing import Any, Optional

from typing_extensions import Annotated
from fastapi.responses import FileResponse

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.enum_class import MakeMode
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.hybrid.hybrid_analyze import doc_analyze as hybrid_doc_analyze
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path

SERVER_URL = "http://127.0.0.1:30000"
DATA_DIR = "data"
IMAGE_MIME_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"]
DOCUMENT_MIME_TYPES = ["application/pdf"]

def do_parse(
    output_dir,  # Output directory for storing parsing results
    pdf_file_names: list[str],  # List of PDF file names to be parsed
    pdf_bytes_list: list[bytes],  # List of PDF bytes to be parsed
    p_lang_list: list[str],  # List of languages for each PDF, default is 'ch' (Chinese)
    backend="hybrid-auto-engine",  # The backend for parsing PDF, default is 'hybrid-auto-engine'
    parse_method="auto",  # The method for parsing PDF, default is 'auto'
    formula_enable=True,  # Enable formula parsing
    table_enable=True,  # Enable table parsing
    server_url=None,  # Server URL for vlm-http-client backend
    f_draw_layout_bbox=True,  # Whether to draw layout bounding boxes
    f_draw_span_bbox=True,  # Whether to draw span bounding boxes
    f_dump_md=True,  # Whether to dump markdown files
    f_dump_middle_json=True,  # Whether to dump middle JSON files
    f_dump_model_output=True,  # Whether to dump model output files
    f_dump_orig_pdf=True,  # Whether to dump original PDF files
    f_dump_content_list=True,  # Whether to dump content list files
    f_make_md_mode=MakeMode.MM_MD,  # The mode for making markdown content, default is MM_MD
    start_page_id=0,  # Start page ID for parsing, default is 0
    end_page_id=None,  # End page ID for parsing, default is None (parse all pages until the end of the document)
):

    if backend == "pipeline":
        for idx, pdf_bytes in enumerate(pdf_bytes_list):
            new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            pdf_bytes_list[idx] = new_pdf_bytes

        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, p_lang_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable)

        for idx, model_list in enumerate(infer_results):
            model_json = copy.deepcopy(model_list)
            pdf_file_name = pdf_file_names[idx]
            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

            images_list = all_image_lists[idx]
            pdf_doc = all_pdf_docs[idx]
            _lang = lang_list[idx]
            _ocr_enable = ocr_enabled_list[idx]
            middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, formula_enable)

            pdf_info = middle_json["pdf_info"]

            pdf_bytes = pdf_bytes_list[idx]
            _process_output(
                pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, model_json, is_pipeline=True
            )
    else:
        f_draw_span_bbox = False

        if backend.startswith("vlm-"):
            backend = backend[4:]

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=False)

            parse_method = "vlm"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                pdf_file_name = pdf_file_names[idx]
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
                middle_json, infer_result = vlm_doc_analyze(pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url)

                pdf_info = middle_json["pdf_info"]

                _process_output(
                    pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                    md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                    f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                    f_make_md_mode, middle_json, infer_result, is_pipeline=False
                )
        elif backend.startswith("hybrid-"):
            backend = backend[7:]

            if backend == "auto-engine":
                backend = get_vlm_engine(inference_engine='auto', is_async=False)

            parse_method = f"hybrid_{parse_method}"
            for idx, pdf_bytes in enumerate(pdf_bytes_list):
                pdf_file_name = pdf_file_names[idx]
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
                local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
                image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
                middle_json, infer_result, _vlm_ocr_enable = hybrid_doc_analyze(
                    pdf_bytes,
                    image_writer=image_writer,
                    backend=backend,
                    parse_method=parse_method,
                    language=p_lang_list[idx],
                    inline_formula_enable=formula_enable,
                    server_url=server_url,
                )

                pdf_info = middle_json["pdf_info"]

                _process_output(
                    pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
                    md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
                    f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                    f_make_md_mode, middle_json, infer_result, is_pipeline=False
                )

def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True
):
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        lang="ch",
        backend="hybrid-auto-engine",
        method="auto",
        server_url=None,
        start_page_id=0,
        end_page_id=None
):
    """
        Parameter description:
        path_list: List of document paths to be parsed, can be PDF or image files.
        output_dir: Output directory for storing parsing results.
        lang: Language option, default is 'ch', optional values include['ch', 'ch_server', 'ch_lite', 'en', 'korean', 'japan', 'chinese_cht', 'ta', 'te', 'ka', 'th', 'el',
                       'latin', 'arabic', 'east_slavic', 'cyrillic', 'devanagari']。
            Input the languages in the pdf (if known) to improve OCR accuracy.  Optional.
            Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'
        backend: the backend for parsing pdf:
            pipeline: More general.
            vlm-auto-engine: High accuracy via local computing power.
            vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers).
            hybrid-auto-engine: Next-generation high accuracy solution via local computing power.
            hybrid-http-client: High accuracy but requires a little local computing power(client suitable for openai-compatible servers).
            Without method specified, hybrid-auto-engine will be used by default.
        method: the method for parsing pdf:
            auto: Automatically determine the method based on the file type.
            txt: Use text extraction method.
            ocr: Use OCR method for image-based PDFs.
            Without method specified, 'auto' will be used by default.
            Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'.
        server_url: When the backend is `http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
        start_page_id: Start page ID for parsing, default is 0
        end_page_id: End page ID for parsing, default is None (parse all pages until the end of the document)
    """
    try:
        file_name_list = []
        pdf_bytes_list = []
        lang_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
            lang_list.append(lang)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            p_lang_list=lang_list,
            backend=backend,
            parse_method=method,
            server_url=server_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)



# Create FastAPI app
app = FastAPI()
mineru_semaphore = asyncio.Semaphore(1)

# Helper Functions
def cleanup_if_file_exceeds_limit(dir_path: str, limit: int = 10) -> bool:
    """
    Remove the directory if it contains more than `limit` items.

    Returns:
        True if the directory was removed, False otherwise.
    """
    if len(os.listdir(dir_path)) > limit:
        shutil.rmtree(dir_path)
        logger.info(f"Cleaned up {dir_path} because it exceeded the limit of {limit} files")
        return True
    return False

def zip_directory(src_dir: str, zip_path: str) -> None:
    """Create a zip archive of the given directory at zip_path."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(src_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, src_dir)
                zipf.write(full_path, rel_path)

# / Helper Functions


@app.post("/analyze-image", response_model=None)
async def analyze_image(
    image: UploadFile = File(..., media_type="image/*"),
    download_dir: Annotated[bool, Form()] = False,
) -> dict[str, Any] | FileResponse:
    """
    Analyze an image using the VLM HTTP client backend.

    Args:
        image: The image file to analyze (JPEG, PNG, GIF, or WebP).
        download_dir: If True, return a zip of the full output directory instead of JSON.

    Returns:
        On success: dict with "success" True, "output_dir", "files" (markdown, model_output,
            content_list, middle_output), and "time_elapsed". If download_dir is True,
            returns a zip FileResponse. On failure: dict with "success" False and "message".
    """
    try:
        if image.content_type not in IMAGE_MIME_TYPES:
            raise ValueError(f"Invalid image MIME type: {image.content_type}")
        
        # Cleanup if there are too many files in the data directory
        if os.path.exists(DATA_DIR):
            cleanup_if_file_exceeds_limit(DATA_DIR)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        # Generate UUID4
        task_id = str(uuid.uuid4())
        
        # Create data directory structure
        task_dir = os.path.join(DATA_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Save the image as data/uuid4/image.png
        image_path = os.path.join(task_dir, "image.png")
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # Use the same folder as output_dir
        output_dir = task_dir
        
        # Call parse_doc with the appropriate backend
        doc_path_list = [Path(image_path)]
        async with mineru_semaphore:
            await asyncio.to_thread(
                parse_doc,
                doc_path_list,
                output_dir,
                backend="vlm-http-client",
                server_url=SERVER_URL
            )
        
        response = {"success": True, "output_dir": output_dir}
        
        output_files_dir = os.path.join(task_dir, "image", "vlm")
        files_to_return = {
            "markdown": "image.md",
            "model_output": "image_model.json",
            "content_list": "image_content_list.json",
            "middle_output": "image_middle.json"
        }
        
        returned_files = {}
        for file_key, filename in files_to_return.items():
            file_path = os.path.join(output_files_dir, filename)
            if os.path.exists(file_path):
                if filename.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        returned_files[file_key] = json.load(f)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        returned_files[file_key] = f.read()
            else:
                returned_files[file_key] = None
        
        response["files"] = returned_files
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        response["time_elapsed"] = round(elapsed_time, 3)  # Round to 3 decimal places (milliseconds precision)

        if download_dir:
            tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp_zip.close()

            zip_directory(task_dir, tmp_zip.name)

            return FileResponse(
                path=tmp_zip.name,
                media_type="application/zip",
                filename=f"{task_id}.zip"
            )
        
        return response

    except Exception as e:
        logger.exception(e)
        return {"success": False, "message": str(e)}


@app.post("/analyze-document", response_model=None)
async def analyze_document(
    document: UploadFile = File(..., media_type="application/pdf", description="The PDF file to analyze."),
    download_dir: Annotated[bool, Form(description="If true, return a zip of the full output directory instead of JSON.")] = False,
    start_page: Annotated[int, Form(description="Start page index (0-based, inclusive). Defaults to 0.")] = 0,
    end_page: Annotated[Optional[int], Form(description="End page index (0-based, inclusive). Defaults to None, which processes all pages to the end.")] = None,
    page_index: Annotated[Optional[int], Form(description="Process only this single page (0-based). When provided, overrides start_page and end_page.")] = None,
) -> dict[str, Any] | FileResponse:
    """
    Analyze a PDF document using the VLM HTTP client backend.

    Args:
        document: The PDF file to analyze.
        download_dir: If True, return a zip of the full output directory instead of JSON.
        start_page: Start page index (0-based, inclusive). Defaults to 0.
        end_page: End page index (0-based, inclusive). Defaults to None (process all pages).
        page_index: If provided, process only this single page (0-based). Overrides start_page and end_page.

    Returns:
        On success: dict with "success" True, "output_dir", "files" (markdown, model_output,
            content_list, middle_output), and "time_elapsed". If download_dir is True,
            returns a zip FileResponse. On failure: dict with "success" False and "message".
    """
    try:
        if document.content_type not in DOCUMENT_MIME_TYPES:
            raise ValueError(f"Invalid document MIME type: {document.content_type}")
        
        # If page_index is provided, override start_page and end_page to target a single page
        if page_index is not None:
            start_page = page_index
            end_page = page_index
        
        # Cleanup if there are too many files in the data directory
        if os.path.exists(DATA_DIR):
            cleanup_if_file_exceeds_limit(DATA_DIR)
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Record start time
        start_time = time.time()
        
        # Generate UUID4
        task_id = str(uuid.uuid4())
        
        # Create data directory structure
        task_dir = os.path.join(DATA_DIR, task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Save the document as data/uuid4/document.pdf
        document_path = os.path.join(task_dir, "document.pdf")
        with open(document_path, "wb") as f:
            content = await document.read()
            f.write(content)
        
        # Use the same folder as output_dir
        output_dir = task_dir
        
        # Call parse_doc with the appropriate backend
        doc_path_list = [Path(document_path)]
        async with mineru_semaphore:
            await asyncio.to_thread(
                parse_doc,
                doc_path_list,
                output_dir,
                backend="vlm-http-client",
                server_url=SERVER_URL,
                start_page_id=start_page,
                end_page_id=end_page,
            )
        
        response = {"success": True, "output_dir": output_dir}
        
        output_files_dir = os.path.join(task_dir, "document", "vlm")
        files_to_return = {
            "markdown": "document.md",
            "model_output": "document_model.json",
            "content_list": "document_content_list.json",
            "middle_output": "document_middle.json"
        }
        
        returned_files = {}
        for file_key, filename in files_to_return.items():
            file_path = os.path.join(output_files_dir, filename)
            if os.path.exists(file_path):
                if filename.endswith(".json"):
                    with open(file_path, "r", encoding="utf-8") as f:
                        returned_files[file_key] = json.load(f)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        returned_files[file_key] = f.read()
            else:
                returned_files[file_key] = None
        
        response["files"] = returned_files
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        response["time_elapsed"] = round(elapsed_time, 3)  # Round to 3 decimal places (milliseconds precision)

        if download_dir:
            tmp_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            tmp_zip.close()

            zip_directory(task_dir, tmp_zip.name)

            return FileResponse(
                path=tmp_zip.name,
                media_type="application/zip",
                filename=f"{task_id}.zip"
            )
        
        return response

    except Exception as e:
        logger.exception(e)
        return {"success": False, "message": str(e)}


# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint to verify if the system is running.
    
#     Returns:
#         dict: Status of the system
#     """
    
#     return {"status": "running", "success": True}

@app.get("/health")
async def health() -> dict[str, Any]:
    """
    Health check: reports API status and whether the VLM backend is reachable.

    Returns:
        dict with "api_status" ("ok"), "backend_status" ("ok" or "down"),
        and optionally "error" if the backend is down.
    """
    try:
        r = requests.get(
            f"{SERVER_URL}/v1/models",
            timeout=2
        )
        if r.status_code != 200:
            raise RuntimeError("models endpoint unhealthy")

        return {
            "api_status": "ok",
            "backend_status": "ok",
            # "models": len(r.json().get("data", []))
        }
    except Exception as e:
        return {
            "api_status": "ok",
            "backend_status": "down",
            "error": str(e)
        }



@app.get("/cleanup")
async def cleanup_data() -> dict[str, Any]:
    """
    Remove all task subdirectories and files under the data directory.

    Returns:
        dict with "success" (bool) and "message" (str).
    """
    try:
        if os.path.exists(DATA_DIR):
            # Remove all contents in the data directory
            for item in os.listdir(DATA_DIR):
                item_path = os.path.join(DATA_DIR, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                else:
                    os.remove(item_path)
            return {"success": True, "message": "Data directory cleaned successfully"}
        else:
            return {"success": True, "message": "Data directory does not exist"}
    except Exception as e:
        logger.exception(e)
        return {"success": False, "message": str(e)}

