"""
build_assets.py — Minify CSS/JS and generate content-hashed bundles.

Usage:
    python scripts/build_assets.py

Outputs:
    static/dist/app.<hash>.min.css
    static/dist/app.<hash>.min.js
    static/dist/manifest.json   (maps original → hashed filenames)
"""

import os
import re
import glob
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DIST_DIR = os.path.join(STATIC_DIR, "dist")


def minify_css(css_text):
    """Basic CSS minification (no external deps)."""
    # Remove comments
    css_text = re.sub(r'/\*.*?\*/', '', css_text, flags=re.DOTALL)
    # Remove unnecessary whitespace
    css_text = re.sub(r'\s+', ' ', css_text)
    # Remove spaces around selectors and properties
    css_text = re.sub(r'\s*([{}:;,>~+])\s*', r'\1', css_text)
    # Remove trailing semicolons before closing braces
    css_text = css_text.replace(';}', '}')
    return css_text.strip()


def minify_js(js_text):
    """Basic JS minification (no external deps)."""
    # Remove single-line comments (but not URLs like http://)
    js_text = re.sub(r'(?<!:)//(?!/).*?$', '', js_text, flags=re.MULTILINE)
    # Remove multi-line comments
    js_text = re.sub(r'/\*.*?\*/', '', js_text, flags=re.DOTALL)
    # Collapse whitespace
    js_text = re.sub(r'\s+', ' ', js_text)
    # Remove spaces around operators (conservative)
    js_text = re.sub(r'\s*([{}();,=<>!&|+\-*/:])\s*', r'\1', js_text)
    # Restore space after keywords
    for kw in ['return', 'const', 'let', 'var', 'if', 'else', 'for',
               'while', 'function', 'new', 'typeof', 'instanceof', 'in',
               'of', 'async', 'await', 'class', 'extends', 'import',
               'export', 'from', 'default', 'throw', 'case']:
        js_text = js_text.replace(f'{kw}{{', f'{kw} {{')
        js_text = js_text.replace(f'{kw}(', f'{kw} (')
    return js_text.strip()


def content_hash(content, length=8):
    """Generate a short hash from content for cache-busting."""
    return hashlib.md5(content.encode()).hexdigest()[:length]


def build():
    """Run the full build pipeline."""
    os.makedirs(DIST_DIR, exist_ok=True)

    # Clean old dist files
    for old in glob.glob(os.path.join(DIST_DIR, "app.*")):
        os.remove(old)

    manifest = {}

    # ── Concatenate & minify CSS ──
    css_files = sorted(glob.glob(os.path.join(STATIC_DIR, "css", "*.css")))
    css_combined = ""
    for f in css_files:
        with open(f) as fh:
            css_combined += fh.read() + "\n"

    if css_combined.strip():
        css_min = minify_css(css_combined)
        css_hash = content_hash(css_min)
        css_filename = f"app.{css_hash}.min.css"
        css_path = os.path.join(DIST_DIR, css_filename)
        with open(css_path, "w") as f:
            f.write(css_min)
        manifest["app.min.css"] = css_filename
        size_kb = len(css_min) / 1024
        orig_kb = len(css_combined) / 1024
        print(f"   CSS: {orig_kb:.1f}KB → {size_kb:.1f}KB ({css_filename})")

    # ── Concatenate & minify JS ──
    js_files = sorted(glob.glob(os.path.join(STATIC_DIR, "js", "*.js")))
    js_combined = ""
    for f in js_files:
        with open(f) as fh:
            js_combined += fh.read() + "\n"

    if js_combined.strip():
        js_min = minify_js(js_combined)
        js_hash = content_hash(js_min)
        js_filename = f"app.{js_hash}.min.js"
        js_path = os.path.join(DIST_DIR, js_filename)
        with open(js_path, "w") as f:
            f.write(js_min)
        manifest["app.min.js"] = js_filename
        size_kb = len(js_min) / 1024
        orig_kb = len(js_combined) / 1024
        print(f"   JS:  {orig_kb:.1f}KB → {size_kb:.1f}KB ({js_filename})")

    # ── Manifest ──
    manifest_path = os.path.join(DIST_DIR, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"   Manifest → {manifest_path}")

    return manifest


if __name__ == "__main__":
    print("🔧 Building assets...")
    build()
    print("✅ Build complete!")
