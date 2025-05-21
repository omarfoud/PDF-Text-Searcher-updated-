import os
import glob
import re
import json
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import shutil
import sys
import subprocess
import time
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text
from whoosh import index, writing
from whoosh.fields import Schema, TEXT, ID, NUMERIC, KEYWORD
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser, OrGroup
from whoosh.highlight import PinpointFragmenter
import nltk
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
nltk.download('punkt', download_dir=r'd:\ir\nltk_data', quiet=True)
nltk.download('stopwords', download_dir=r'd:\ir\nltk_data', quiet=True)
nltk.download('averaged_perceptron_tagger', download_dir=r'd:\ir\nltk_data', quiet=True)
nltk.download('wordnet', download_dir=r'd:\ir\nltk_data', quiet=True)
nltk.download('omw-1.4', download_dir=r'd:\ir\nltk_data', quiet=True)
_NLTK_DATA_SUBDIR = "nltk_data"
def _configure_nltk_data_path():
    application_root_path = ""
    if getattr(sys, 'frozen', False):
        application_root_path = os.path.dirname(sys.executable)
    else:
        try:
            application_root_path = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            application_root_path = os.getcwd()
    local_nltk_path = os.path.join(application_root_path, _NLTK_DATA_SUBDIR)
    if os.path.isdir(local_nltk_path):
        if local_nltk_path not in nltk.data.path:
            nltk.data.path.insert(0, local_nltk_path)
            print(f"INFO: NLTK will search for data in local directory: {local_nltk_path}")
    return application_root_path, local_nltk_path

def check_nltk_resources():
    application_root_path, local_nltk_full_path = _configure_nltk_data_path()
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    all_found = True
    missing_resources_details = []
    print("INFO: Checking for required NLTK resources...")
    for resource_path, resource_id in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            all_found = False
            error_message_line = (
                f"- '{resource_id}' (expected at '{resource_path}')"
            )
            missing_resources_details.append(error_message_line)

    if not all_found:
        print("-" * 60)
        print("ERROR: One or more required NLTK resources were not found:")
        for detail in missing_resources_details:
            print(f"  {detail}")
        print("\nTo resolve this, download the missing resources into a local 'nltk_data' directory:")
        print(f"  1. Create a directory named '{_NLTK_DATA_SUBDIR}' in the application's directory:")
        print(f"     '{application_root_path}'")
        print(f"  2. In a Python interpreter, run the following for each missing resource ID (e.g., 'punkt'):")
        print(f"     >>> import nltk")
        print(f"     >>> nltk.download('RESOURCE_ID_HERE', download_dir=r'{local_nltk_full_path}')")
        print(f"     (Replace RESOURCE_ID_HERE with the actual ID like 'punkt', 'stopwords', etc.)")
        print(f"     (Ensure the `download_dir` path '{local_nltk_full_path}' is correct.)")
        print("\nAlternatively, if NLTK resources are installed globally, ensure NLTK's data path includes their location.")
        print(f"Current NLTK search paths: {nltk.data.path}")
        print("-" * 60)
        print("WARNING: Text normalization will be basic (lowercase only) until all NLTK resources are available.")
        print("-" * 60)
    else:
        print("INFO: All required NLTK resources are available.")
    return all_found

NLTK_READY = check_nltk_resources()

# --- Conditional NLTK Import & Fallbacks ---
if NLTK_READY:
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords as nltk_stopwords_corpus
    from nltk import pos_tag, word_tokenize
    lemmatizer = WordNetLemmatizer()
    try:
        _stop_words_set = set(nltk_stopwords_corpus.words('english'))
    except LookupError:
        print("ERROR: Could not load NLTK stopwords even though NLTK_READY was initially true. Check 'stopwords' resource.")
        NLTK_READY = False
        _stop_words_set = set()
else:
    lemmatizer = None
    _stop_words_set = set()

if not _stop_words_set and not NLTK_READY:
    print("INFO: NLTK resources unavailable. Using a basic internal list of stopwords.")
    _stop_words_set = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

# --- Text Preprocessing Utilities ---
def get_wordnet_pos(treebank_tag):
    if not NLTK_READY: return 'n'
    if treebank_tag.startswith('J'): return 'a'
    elif treebank_tag.startswith('V'): return 'v'
    elif treebank_tag.startswith('R'): return 'r'
    else: return 'n'

def normalize_text(text, preserve_wildcards=False):
    if not NLTK_READY or not lemmatizer:
        return str(text).lower() if text else ""

    if not isinstance(text, str): text = str(text)
    text = text.lower()
    
    # Preserve wildcard characters if preserve_wildcards is True
    if preserve_wildcards:
        # Replace wildcards with temporary placeholders
        text = text.replace('*', '___WILDCARD_STAR___')
        text = text.replace('?', '___WILDCARD_QMARK___')
    
    # Remove other special characters except spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    
    # Restore wildcards if they were preserved
    if preserve_wildcards:
        text = text.replace('___WILDCARD_STAR___', '*')
        text = text.replace('___WILDCARD_QMARK___', '?')
    
    # Skip tokenization for wildcard queries to preserve the wildcards
    if preserve_wildcards and ('*' in text or '?' in text):
        return text
        
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words_set and len(t) > 1 and not t.isdigit()]

    tagged = pos_tag(tokens)
    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)) for token, pos in tagged]
    return " ".join(lemmas)

# --- Whoosh Index Schema Definition ---
INDEX_DIR = 'indexdir'
content_analyzer = StemmingAnalyzer() if NLTK_READY else None
title_analyzer = StemmingAnalyzer() if NLTK_READY else None

schema = Schema(
    path=ID(stored=True, unique=True),
    filetype=KEYWORD(stored=True),
    content=TEXT(analyzer=content_analyzer, phrase=True, stored=False),
    pagenum=NUMERIC(stored=True),
    rownum=NUMERIC(stored=True),
    sheetname=ID(stored=True),
    jsonkey=ID(stored=True),
    title=TEXT(stored=True, analyzer=title_analyzer)
)

# --- File-Specific Indexing Functions ---
def index_pdf(path, writer, app_instance):
    try:
        full_text = extract_text(path)
        pages = full_text.split('\f')
        for i, page_content in enumerate(pages, 1):
            if app_instance.stop_indexing_flag: break
            normalized_content = normalize_text(page_content)
            if normalized_content:
                writer.update_document(
                    path=f"{path}::page::{i}", filetype='pdf', content=normalized_content, pagenum=i,
                    title=os.path.basename(path)
                )
    except Exception as e:
        print(f"Error indexing PDF {path} (page {i if 'i' in locals() else 'N/A'}): {e}")
        app_instance.update_status_from_thread(f"Error PDF: {os.path.basename(path)} - {e}", is_error=True)

def index_txt(path, writer, app_instance):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        normalized_content = normalize_text(content)
        if normalized_content:
            writer.update_document(path=path, filetype='txt', content=normalized_content, title=os.path.basename(path))
    except Exception as e:
        print(f"Error indexing TXT {path}: {e}")
        app_instance.update_status_from_thread(f"Error TXT: {os.path.basename(path)} - {e}", is_error=True)

def index_csv(path, writer, app_instance):
    try:
        # Try reading with default encoding first
        try:
            df = pd.read_csv(path, on_bad_lines='skip', lineterminator='\n')
        except UnicodeDecodeError:
            # If UTF-8 fails, try with different encodings
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(path, on_bad_lines='skip', lineterminator='\n', encoding=encoding)
                    break
                except:
                    continue
            if df is None:
                raise ValueError(f"Could not read CSV file with any of the following encodings: {', '.join(encodings)}")
        
        # Index each row with column names as context
        for i, row in df.iterrows():
            if app_instance.stop_indexing_flag: 
                break
                
            # Create a more informative content string that includes column names
            row_content = []
            for col_name, value in row.items():
                if pd.notna(value):
                    row_content.append(f"{col_name}: {value}")
            
            if not row_content:
                continue
                
            combined = ' | '.join(row_content)
            normalized_content = normalize_text(combined)
            
            if normalized_content:
                writer.update_document(
                    path=f"{path}::row::{i+1}", 
                    filetype='csv', 
                    content=normalized_content,
                    rownum=i + 1, 
                    sheetname=os.path.basename(path), 
                    title=os.path.basename(path)
                )
                
        # Also index the header/column names as a separate document
        header_content = ' | '.join(df.columns.tolist())
        normalized_header = normalize_text(header_content)
        if normalized_header:
            writer.update_document(
                path=f"{path}::header",
                filetype='csv_header',
                content=normalized_header,
                sheetname=os.path.basename(path),
                title=f"{os.path.basename(path)} (Columns)"
            )
    except Exception as e:
        print(f"Error indexing CSV {path} (row {i+1 if 'i' in locals() else 'N/A'}): {e}")
        app_instance.update_status_from_thread(f"Error CSV: {os.path.basename(path)} - {e}", is_error=True)

def index_excel(path, writer, app_instance):
    try:
        xls = pd.ExcelFile(path)
        for sheet_name in xls.sheet_names:
            if app_instance.stop_indexing_flag: break
            df = xls.parse(sheet_name)
            for i, row in df.iterrows():
                if app_instance.stop_indexing_flag: break
                combined_parts = [str(val) for val in row.values if pd.notna(val)]
                if not combined_parts: continue
                combined = ' '.join(combined_parts)
                normalized_content = normalize_text(combined)
                if normalized_content:
                    writer.update_document(path=f"{path}::sheet::{sheet_name}::row::{i+1}", filetype='excel', content=normalized_content,
                                        rownum=i + 1, sheetname=sheet_name, title=os.path.basename(path))
    except Exception as e:
        print(f"Error indexing Excel {path} (Sheet: {sheet_name if 'sheet_name' in locals() else 'N/A'}, Row: {i+1 if 'i' in locals() else 'N/A'}): {e}")
        app_instance.update_status_from_thread(f"Error Excel: {os.path.basename(path)} - {e}", is_error=True)

def flatten_json_for_indexing(y, current_path_parts=None):
    if current_path_parts is None:
        current_path_parts = []
    if isinstance(y, dict):
        dict_texts = []
        for k, v in y.items():
            for sub_path_parts, text_val in flatten_json_for_indexing(v, current_path_parts + [str(k)]):
                yield sub_path_parts, text_val
                if isinstance(text_val, str):
                    dict_texts.append(text_val)
        if dict_texts:
             yield ".".join(current_path_parts) if current_path_parts else "root", " ".join(dict_texts)
    elif isinstance(y, list):
        list_texts = []
        for i, item in enumerate(y):
            for sub_path_parts, text_val in flatten_json_for_indexing(item, current_path_parts + [str(i)]):
                yield sub_path_parts, text_val
                if isinstance(text_val, str):
                    list_texts.append(text_val)
        if list_texts:
            yield ".".join(current_path_parts) if current_path_parts else "root_list", " ".join(list_texts)
    else:
        yield ".".join(current_path_parts), str(y)

def index_json(path, writer, app_instance):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        full_json_text = normalize_text(json.dumps(data))
        if full_json_text:
            writer.update_document(path=path, filetype='json', content=full_json_text,
                                title=os.path.basename(path), jsonkey='_full_object_')
        for key_path_str, text_content in flatten_json_for_indexing(data):
            if app_instance.stop_indexing_flag: break
            normalized_val_text = normalize_text(text_content)
            if normalized_val_text and key_path_str:
                whoosh_path = f"{path}::json_key::{key_path_str}"
                writer.update_document(path=whoosh_path, filetype='json', content=normalized_val_text,
                                    jsonkey=key_path_str, title=os.path.basename(path))
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error in {path}: {e}")
        app_instance.update_status_from_thread(f"Error JSON Decode: {os.path.basename(path)} - {e}", is_error=True)
    except Exception as e:
        print(f"Error indexing JSON {path} (Key: {key_path_str if 'key_path_str' in locals() else 'N/A'}): {e}")
        app_instance.update_status_from_thread(f"Error JSON: {os.path.basename(path)} - {e}", is_error=True)

def index_html(path_or_url, writer, app_instance):
    try:
        is_url = path_or_url.lower().startswith('http')
        doc_title = path_or_url
        if is_url:
            try:
                req = Request(path_or_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(req, timeout=15) as response:
                    html_content = response.read()
                    content_type = response.info().get_content_type()
                    if 'html' not in content_type.lower():
                        print(f"Skipping non-HTML URL: {path_or_url} (Type: {content_type})")
                        app_instance.update_status_from_thread(f"Skipped (not HTML): {path_or_url[:50]}...", is_error=True)
                        return
            except (URLError, HTTPError) as url_err:
                print(f"URL Error accessing {path_or_url}: {url_err}")
                app_instance.update_status_from_thread(f"URL Access Error: {path_or_url[:50]}... - {url_err}", is_error=True)
                return
        else:
            try:
                with open(path_or_url, 'rb') as f:
                    html_content = f.read()
                doc_title = os.path.basename(path_or_url)
            except IOError as file_err:
                print(f"File Error reading {path_or_url}: {file_err}")
                app_instance.update_status_from_thread(f"File Read Error: {path_or_url[:50]}... - {file_err}", is_error=True)
                return
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            if soup.title and soup.title.string:
                doc_title = soup.title.string.strip()
            for R_tag in soup(['script', 'style', 'head', 'meta', 'nav', 'footer', 'aside']):
                R_tag.extract()
            main_content_el = soup.find('main') or soup.find('article') or soup.find('div', role='main')
            if main_content_el:
                text = main_content_el.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            normalized_content = normalize_text(text)
            normalized_title = normalize_text(doc_title)
            if normalized_content:
                writer.update_document(path=path_or_url, filetype='html', content=normalized_content, title=normalized_title or doc_title)
            else:
                print(f"No content extracted from {path_or_url}")
                app_instance.update_status_from_thread(f"No content: {path_or_url[:50]}...", is_error=True)
        except Exception as parse_err:
            print(f"Parsing Error for {path_or_url}: {parse_err}")
            app_instance.update_status_from_thread(f"HTML Parse Error: {path_or_url[:50]}... - {parse_err}", is_error=True)
    except Exception as e:
        print(f"Unexpected Error indexing HTML {path_or_url}: {e}")
        app_instance.update_status_from_thread(f"Unexpected HTML Error: {path_or_url[:50]}... - {e}", is_error=True)

# --- Directory Ingestion Runner ---
def ingest_directory_content_runner(directory, formats_to_index, writer_instance, app_instance):
    file_handlers = {
        'pdf': index_pdf, 'txt': index_txt, 'csv': index_csv,
        'xlsx': index_excel, 'json': index_json, 'html': index_html, 'htm': index_html
    }
    files_to_process_list = []
    for fmt in formats_to_index:
        paths = glob.glob(os.path.join(directory, f"**/*.{fmt}"), recursive=True)
        files_to_process_list.extend([(fmt, path) for path in paths])
    total_files_to_process = len(files_to_process_list)
    if not files_to_process_list:
        app_instance.update_status_from_thread("No files found for selected formats.")
        return 0
    processed_count = 0
    for fmt, path in files_to_process_list:
        if app_instance.stop_indexing_flag:
            app_instance.update_status_from_thread("Indexing stopped by user.")
            break
        base_name = os.path.basename(path)
        app_instance.update_status_from_thread(f"Idx [{processed_count+1}/{total_files_to_process}]: {base_name}")
        if fmt in file_handlers:
            file_handlers[fmt](path, writer_instance, app_instance)
        processed_count += 1
        app_instance.update_progress_from_thread((processed_count / total_files_to_process) * 100)
    return processed_count

# --- Main Application GUI (Tkinter) ---
class SearchApp:
    def __init__(self, master_root):
        self.root = master_root
        self.root.title('FOUDA WEB BROWSER')
        self.root.geometry("900x700")
        self.formats = {'pdf', 'txt', 'csv', 'xlsx', 'json', 'html', 'htm'}
        self.directory = tk.StringVar(value=os.getcwd())
        self.ix = None
        self.stop_indexing_flag = False
        self.last_error_time = 0
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("Error.TLabel", foreground="red")
        self.paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.controls_frame = ttk.Labelframe(self.paned_window, text="Controls", padding=10)
        self.paned_window.add(self.controls_frame, weight=1)
        ttk.Button(self.controls_frame, text='Select & Index Directory', command=self.select_dir_and_index).grid(row=0, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.dir_label_var = tk.StringVar(value="No directory selected.")
        dir_display_label = ttk.Label(self.controls_frame, textvariable=self.dir_label_var, wraplength=220)
        dir_display_label.grid(row=1, column=0, columnspan=3, padx=5, pady=2, sticky="ew")
        
        # URL Indexing UI Block
        ttk.Label(self.controls_frame, text="Index URL:").grid(row=2, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
        self.url_entry = ttk.Entry(self.controls_frame, width=30)
        self.url_entry.grid(row=3, column=0, columnspan=2, padx=5, pady=2, sticky="ew")
        self.url_entry.bind("<Return>", lambda e: self.index_single_url())
        self.url_button = ttk.Button(self.controls_frame, text="Index URL", command=self.index_single_url)
        self.url_button.grid(row=3, column=2, padx=5, pady=2, sticky="ew")
        
        ttk.Label(self.controls_frame, text="File types:").grid(row=4, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
        self.check_vars = {f: tk.IntVar(value=1) for f in sorted(list(self.formats))}
        current_row, current_col = 5, 0
        for f_format in sorted(list(self.formats)):
            cb = ttk.Checkbutton(self.controls_frame, text=f_format.upper(), variable=self.check_vars[f_format])
            cb.grid(row=current_row, column=current_col, padx=2, pady=2, sticky="w")
            current_col += 1
            if current_col >= 3: current_col = 0; current_row += 1
        self.reindex_button = ttk.Button(self.controls_frame, text="Re-Index Current Directory", command=self.reindex_current_directory, state="disabled")
        self.reindex_button.grid(row=current_row+1, column=0, columnspan=3, padx=5, pady=10, sticky="ew")
        ttk.Label(self.controls_frame, text="Search Query:").grid(row=current_row+2, column=0, columnspan=3, sticky="w", padx=5, pady=(10,0))
        self.query_entry = ttk.Entry(self.controls_frame, width=30)
        self.query_entry.grid(row=current_row+3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.query_entry.bind("<Return>", self.perform_search_threaded)
        self.search_button = ttk.Button(self.controls_frame, text='Search', command=self.perform_search_threaded)
        self.search_button.grid(row=current_row+3, column=2, padx=5, pady=5, sticky="ew")
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)
        self.results_frame = ttk.Labelframe(self.paned_window, text="Search Results", padding=10)
        self.paned_window.add(self.results_frame, weight=4)
        columns = ('file', 'type', 'title', 'location', 'score', 'snippet')
        self.tree = ttk.Treeview(self.results_frame, columns=columns, show='headings')
        col_widths = {'file': 150, 'type': 50, 'title': 150, 'location': 120, 'score': 50, 'snippet': 300}
        for col_name in columns:
            self.tree.heading(col_name, text=col_name.capitalize(), command=lambda c=col_name: self.sort_treeview_column(c, False))
            self.tree.column(col_name, width=col_widths.get(col_name, 100), anchor='w')
        self.tree.column('score', anchor='center')
        vsb = ttk.Scrollbar(self.results_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(self.results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self.tree.pack(fill='both', expand=True)
        self.tree.bind("<Double-1>", self.open_selected_file_from_tree)
        self.bottom_frame = ttk.Frame(self.root, padding=5)
        self.bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label_var = tk.StringVar(value="Status: Ready.")
        self.status_label = ttk.Label(self.bottom_frame, textvariable=self.status_label_var, relief="sunken", anchor="w")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
        self.progress_bar = ttk.Progressbar(self.bottom_frame, orient="horizontal", length=200, mode="determinate")
        self.progress_bar.pack(side=tk.LEFT, padx=(0,5))
        self.stop_button = ttk.Button(self.bottom_frame, text="Stop Indexing", command=self.signal_stop_indexing, state="disabled")
        self.stop_button.pack(side=tk.LEFT)
        self.check_existing_index()
        if not NLTK_READY:
            self.root.after(500, lambda: messagebox.showwarning("NLTK Warning",
                "NLTK resources are not fully available or configured correctly.\n"
                "Text normalization will be basic (lowercase only).\n"
                "Advanced features like stemming and lemmatization will be disabled.\n"
                "Please check the console output for instructions on how to set up NLTK resources."
            ))

    def sort_treeview_column(self, col, reverse):
        # Sort treeview column when header clicked
        l = [(self.tree.set(k, col), k) for k in self.tree.get_children('')]
        try:
            l.sort(key=lambda t: float(t[0]) if t[0] else -1, reverse=reverse)
        except ValueError:
            l.sort(key=lambda t: str(t[0]).lower(), reverse=reverse)
        for index, (val, k) in enumerate(l):
            self.tree.move(k, '', index)
        self.tree.heading(col, command=lambda: self.sort_treeview_column(col, not reverse))

    def check_existing_index(self):
        # Check for and load existing Whoosh index
        if os.path.exists(INDEX_DIR) and index.exists_in(INDEX_DIR):
            try:
                self.ix = index.open_dir(INDEX_DIR)
                self.update_status(f"Existing index loaded. {self.ix.doc_count()} docs. Ready.")
                self.dir_label_var.set(self.format_dir_label(self.directory.get()))
                self.reindex_button.config(state="normal")
            except Exception as e:
                self.update_status(f"Error opening existing index: {e}. Consider re-indexing.", is_error=True)
                self.ix = None
                if messagebox.askyesno("Index Error", f"Error opening existing index: {e}\nThis might be due to a schema change or corruption.\nDo you want to delete the old index directory and start fresh?"):
                    try:
                        shutil.rmtree(INDEX_DIR)
                        self.update_status("Old index directory deleted. Please select a directory to index.")
                    except Exception as del_e:
                        self.update_status(f"Error deleting index directory: {del_e}", is_error=True)
                self.reindex_button.config(state="disabled")
        else:
            self.update_status("No existing index. Please select a directory to build one.")
            self.reindex_button.config(state="disabled")

    def format_dir_label(self, dir_path):
        # Format directory path for display
        if not dir_path: return "No directory selected."
        max_len = 35
        if len(dir_path) <= max_len: return dir_path
        return "..." + dir_path[-(max_len-3):]

    # Status bar and progress bar updates (thread-safe)
    def update_status(self, message, is_error=False):
        current_time = time.time()
        if is_error and (current_time - self.last_error_time < 0.5):
            return
        self.status_label_var.set(f"Status: {message}")
        if is_error:
            self.status_label.config(style="Error.TLabel")
            self.last_error_time = current_time
        else:
            self.status_label.config(style="TLabel")
        self.root.update_idletasks()

    def update_status_from_thread(self, message, is_error=False):
        self.root.after(0, self.update_status, message, is_error)

    def update_progress_from_thread(self, value):
        self.root.after(0, self.progress_bar.config, {'value': value})

    def signal_stop_indexing(self):
        # Signal to stop the indexing thread
        self.stop_indexing_flag = True
        self.update_status("Stop signal sent. Finishing current file...")
        self.stop_button.config(state="disabled")

    def _set_ui_state_busy(self, is_busy, task_name="Processing"):
        # Enable/disable UI controls during operations
        control_state = "disabled" if is_busy else "normal"
        for child in self.controls_frame.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Entry, ttk.Checkbutton)):
                 if child not in [self.stop_button]:
                    child.config(state=control_state)
        if not is_busy and self.directory.get() and os.path.isdir(self.directory.get()):
            self.reindex_button.config(state="normal")
        elif is_busy:
            self.reindex_button.config(state="disabled")
        self.stop_button.config(state="normal" if is_busy and task_name == "Indexing" else "disabled")
        if not is_busy:
            self.progress_bar['value'] = 0
            if task_name == "Indexing": self.stop_indexing_flag = False

    # UI Actions: Select directory, Re-index, Start indexing
    def index_single_url(self):
        """Handle indexing of a single URL from the URL entry field."""
        url = self.url_entry.get().strip()
        if not url:
            self.update_status("Please enter a URL to index.", is_error=True)
            return
            
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'http://' + url
            
        self._set_ui_state_busy(True, "Indexing URL")
        self.url_button.config(state="disabled")
        self.url_entry.config(state="disabled")
        
        def worker():
            try:
                if not os.path.exists(INDEX_DIR):
                    os.makedirs(INDEX_DIR)
                
                # Create or open the index
                if not index.exists_in(INDEX_DIR):
                    ix = index.create_in(INDEX_DIR, schema)
                else:
                    ix = index.open_dir(INDEX_DIR)
                
                writer = ix.writer()
                
                # Index the URL using the existing HTML indexer
                index_html(url, writer, self)
                
                writer.commit(merge=True)
                self.root.after(0, lambda: self.update_status(f"Successfully indexed URL: {url}"))
            except Exception as e:
                self.root.after(0, lambda: self.update_status(f"Error indexing URL: {e}", is_error=True))
            finally:
                self.root.after(0, self._enable_url_ui)
        
        # Start the indexing in a separate thread
        threading.Thread(target=worker, daemon=True).start()
    
    def _enable_url_ui(self):
        """Re-enable the URL UI elements after indexing is done or failed."""
        self._set_ui_state_busy(False)
        self.url_button.config(state="normal")
        self.url_entry.config(state="normal")
        self.url_entry.focus_set()
    
    def select_dir_and_index(self):
        selected_dir = filedialog.askdirectory(initialdir=self.directory.get() or os.path.expanduser("~"))
        if not selected_dir: return
        self.directory.set(selected_dir)
        self.dir_label_var.set(self.format_dir_label(selected_dir))
        self.reindex_button.config(state="normal")
        self.start_indexing_thread()

    def reindex_current_directory(self):
        current_dir = self.directory.get()
        if not current_dir or not os.path.isdir(current_dir):
            messagebox.showwarning("Re-Index", "No valid directory selected or current directory is not accessible.")
            self.select_dir_and_index()
            return
        if messagebox.askyesno("Re-Index", f"Re-index '{self.format_dir_label(current_dir)}'?\nThis will delete the current index if it exists."):
            self.start_indexing_thread()

    def start_indexing_thread(self):
        dir_to_index = self.directory.get()
        if not dir_to_index or not os.path.isdir(dir_to_index):
            messagebox.showerror("Error", "No valid directory specified for indexing.")
            return
        self._set_ui_state_busy(True, "Indexing")
        self.update_status(f"Starting indexing for: {self.format_dir_label(dir_to_index)}")
        self.progress_bar['value'] = 0
        self.tree.delete(*self.tree.get_children())
        formats_to_index = [f for f, v in self.check_vars.items() if v.get()]
        if not formats_to_index:
            messagebox.showinfo("Indexing", "No file types selected.")
            self._set_ui_state_busy(False, "Indexing")
            return
        thread = threading.Thread(target=self._ingest_worker_thread_target, args=(dir_to_index, formats_to_index), daemon=True)
        thread.start()

    def _ingest_worker_thread_target(self, directory_to_index, formats):
        # Worker thread function for indexing
        self.stop_indexing_flag = False
        num_files_processed = 0
        try:
            if os.path.exists(INDEX_DIR):
                self.update_status_from_thread(f"Clearing old index: {INDEX_DIR}")
                try:
                    shutil.rmtree(INDEX_DIR)
                except Exception as e:
                    self.update_status_from_thread(f"Error clearing old index: {e}. Proceeding...", is_error=True)
            os.makedirs(INDEX_DIR, exist_ok=True)

            current_content_analyzer = StemmingAnalyzer() if NLTK_READY else None
            current_title_analyzer = StemmingAnalyzer() if NLTK_READY else None
            current_schema = Schema(
                path=ID(stored=True, unique=True),
                filetype=KEYWORD(stored=True),
                content=TEXT(analyzer=current_content_analyzer, phrase=True, stored=False),
                pagenum=NUMERIC(stored=True),
                rownum=NUMERIC(stored=True),
                sheetname=ID(stored=True),
                jsonkey=ID(stored=True),
                title=TEXT(stored=True, analyzer=current_title_analyzer)
            )
            current_ix_local = index.create_in(INDEX_DIR, current_schema)
            writer = current_ix_local.writer(limitmb=128)
            try:
                num_files_processed = ingest_directory_content_runner(directory_to_index, formats, writer, self)
            finally:
                if not self.stop_indexing_flag:
                    self.update_status_from_thread("Committing index...")
                    writer.commit()
                else:
                    writer.cancel()
                    self.update_status_from_thread("Indexing stopped. Index commit cancelled.")

            if not self.stop_indexing_flag:
                self.ix = current_ix_local
                self.root.after(0, lambda n=num_files_processed: messagebox.showinfo("Indexing Complete", f"Indexing finished.\n{n} files processed."))
                self.update_status_from_thread(f"Index ready. {self.ix.doc_count()} docs. {num_files_processed} files processed.")
            else:
                 self.ix = None
                 self.update_status_from_thread("Indexing was stopped. Index may be incomplete or unusable.")
        except Exception as e:
            self.ix = None
            self.update_status_from_thread(f"Critical Indexing Error: {e}", is_error=True)
            self.root.after(0, lambda err=e: messagebox.showerror("Indexing Error", f"A critical error occurred during indexing: {str(err)}"))
        finally:
            self.root.after(0, self._set_ui_state_busy, False, "Indexing")

    # UI Action & Worker thread function for searching
    def perform_search_threaded(self, event=None):
        query_text = self.query_entry.get()
        if not query_text.strip():
            messagebox.showinfo("Search", "Please enter a search query.")
            return
        if not self.ix:
            if os.path.exists(INDEX_DIR) and index.exists_in(INDEX_DIR):
                try:
                    self.ix = index.open_dir(INDEX_DIR)
                    self.update_status(f"Opened existing index. {self.ix.doc_count()} docs.")
                except Exception as e:
                    messagebox.showerror("Search Error", f"Could not open index: {e}\nThis may be due to schema changes or corruption. Please re-index.")
                    self.ix = None
                    return
            else:
                 messagebox.showerror("Search Error", "No index found. Please select a directory and index first.")
                 return
        if not self.ix:
            messagebox.showerror("Search Error", "Index is not available. Please re-index.")
            return
        self._set_ui_state_busy(True, "Searching")
        self.update_status(f"Searching for: '{query_text}'")
        self.progress_bar.config(mode='indeterminate')
        self.progress_bar.start()
        thread = threading.Thread(target=self._search_worker_thread_target, args=(query_text,), daemon=True)
        thread.start()

    def _search_worker_thread_target(self, query_text):
        hits_data_for_gui = []
        seen_files = set()
        try:
            with self.ix.searcher() as searcher:
                # Check if the query contains wildcards
                contains_wildcards = '*' in query_text or '?' in query_text
                
                # Create the appropriate query parser
                if contains_wildcards:
                    # For wildcard queries, we'll use a different approach
                    from whoosh.query import Wildcard, Or
                    from whoosh.qparser import QueryParser
                    
                    # Split the query into terms
                    terms = query_text.split()
                    queries = []
                    
                    for term in terms:
                        # If the term contains wildcards, create a Wildcard query
                        if '*' in term or '?' in term:
                            # Convert SQL-like wildcards to Whoosh wildcards
                            # (Whoosh uses * for multiple chars, ? for single char - same as SQL)
                            queries.append(Wildcard("content", term.lower()))
                        else:
                            # For non-wildcard terms, use a standard term query
                            queries.append(QueryParser("content", self.ix.schema).parse(term.lower()))
                    
                    # Combine queries with OR for any term matching
                    if queries:
                        combined_query = Or(queries) if len(queries) > 1 else queries[0]
                        results = searcher.search(combined_query, limit=None)
                    else:
                        results = []
                else:
                    # Standard search for non-wildcard queries
                    parser = MultifieldParser(
                        ["title^1.5", "content"],
                        schema=self.ix.schema,
                        group=OrGroup
                    )
                    
                    normalized_query = normalize_text(query_text)
                    if not normalized_query:
                        self.root.after(0, lambda: self.tree.delete(*self.tree.get_children()))
                        self.update_status_from_thread("Query empty after normalization.")
                        self.root.after(0, lambda: messagebox.showinfo("Search", "Query too common or empty after processing."))
                        return
                    
                    results = searcher.search(parser.parse(normalized_query), limit=None, terms=True)
                
                frag = PinpointFragmenter(maxchars=200, surround=40, autotrim=True)

                for hit in results:
                    raw_path = hit.get('path', 'N/A').split("::")[0]
                    # Skip if we've already added this file
                    if raw_path in seen_files:
                        continue
                    seen_files.add(raw_path)

                    display_filename = os.path.basename(raw_path)
                    filetype = hit.get('filetype', '')
                    title_val = hit.get('title', display_filename)

                    # Build a simple “location” string
                    loc_parts = []
                    if hit.get('sheetname'): loc_parts.append(f"Sht:{hit['sheetname']}")
                    if hit.get('pagenum'):   loc_parts.append(f"Pg:{hit['pagenum']}")
                    if hit.get('rownum'):    loc_parts.append(f"Rw:{hit['rownum']}")
                    if hit.get('jsonkey') and hit['jsonkey'] != '_full_object_':
                        jk = hit['jsonkey']
                        loc_parts.append(f"Key:{jk[:30]}{'...' if len(jk)>30 else ''}")
                    location = ', '.join(loc_parts) or '-'

                    score = round(hit.score, 2)

                    # Highlight and then truncate to 200 chars max
                    snippet = ""
                    try:
                        raw_snip = hit.highlights("content", fragmenter=frag, top=1)
                        snippet = BeautifulSoup(raw_snip, "html.parser").get_text().strip()
                    except Exception:
                        snippet = title_val
                    if len(snippet) > 200:
                        snippet = snippet[:197] + "..."

                    hits_data_for_gui.append((
                        display_filename,
                        filetype,
                        title_val,
                        location,
                        score,
                        snippet,
                        raw_path
                    ))

        except Exception as e:
            self.update_status_from_thread(f"Search Error: {e}", is_error=True)
            self.root.after(0, lambda err=e: messagebox.showerror("Search Error", f"An error occurred during search: {err}"))
        finally:
            # Update GUI with deduped, truncated results
            self.root.after(0, self.display_search_results_in_gui, hits_data_for_gui, query_text)
            self.root.after(0, self.progress_bar.stop)
            self.root.after(0, self.progress_bar.config, {'mode':'determinate', 'value': 0})
            self.root.after(0, self._set_ui_state_busy, False, "Searching")


    def display_search_results_in_gui(self, hits_data, query_text):
        # Update the results treeview
        self.tree.delete(*self.tree.get_children())
        if not hits_data:
            self.update_status(f"No results found for: '{query_text}'")
            return
        for item_data in hits_data:
            self.tree.insert('', 'end', values=item_data[:-1], iid=item_data[-1])
        self.update_status(f"Found {len(hits_data)} results for: '{query_text}'")

    def open_selected_file_from_tree(self, event):
        # Open the selected file/URL on double-click
        selected_iid = self.tree.focus()
        if not selected_iid: return
        file_path_to_open = selected_iid
        if file_path_to_open.lower().startswith('http'):
            try:
                import webbrowser
                webbrowser.open_new_tab(file_path_to_open)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open URL '{file_path_to_open}': {e}")
            return
        if not os.path.exists(file_path_to_open):
            messagebox.showerror("Error", f"File not found: {file_path_to_open}\nIt may have been moved or deleted.")
            return
        try:
            if sys.platform == "win32": os.startfile(os.path.normpath(file_path_to_open))
            elif sys.platform == "darwin": subprocess.call(["open", file_path_to_open])
            else: subprocess.call(["xdg-open", file_path_to_open])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file '{file_path_to_open}': {e}")


if __name__ == '__main__':
    root = tk.Tk()
    app = SearchApp(root)
    root.mainloop()