project = "PySIFT"
author = "Sivakumar K.S."
copyright = "2026, Sivakumar K.S., IIT Madras"
release = "0.1.4"

extensions = [
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinx_copybutton",
]

html_theme = "furo"
html_title = "PySIFT Documentation"
html_baseurl = "https://pysift.readthedocs.io/en/latest/"

html_theme_options = {
    "source_repository": "https://github.com/SivaIITM/PySIFT",
    "source_branch": "master",
    "source_directory": "rtd/",
    "light_css_variables": {
        "color-brand-primary": "#27ae60",
        "color-brand-content": "#27ae60",
    },
}

# OpenGraph meta
ogp_site_url = "https://pysift.readthedocs.io/en/latest/"
ogp_site_name = "PySIFT Documentation"
ogp_description_length = 200

# Sitemap
sitemap_url_scheme = "{link}"

exclude_patterns = ["_build"]
