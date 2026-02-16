# Analytics & Dashboard Research

## Problem Statement

Users want visual insights into their Reddit saving behavior: what topics dominate,
how interests evolve, which authors they save most, and content type breakdown.

## Analytics Features

### Core Metrics

| Metric | Description | Visualization |
|--------|-------------|---------------|
| Saves over time | Timeline of saves (daily/weekly/monthly) | Line chart |
| Top subreddits | Most-saved subreddits (top 20) | Bar chart / treemap |
| Content type distribution | Text vs link vs media vs gallery | Pie chart |
| Top authors | Authors you save from most | Horizontal bar |
| Score distribution | Upvote counts of saved posts | Histogram |
| Word count distribution | Short vs long posts | Histogram |
| Posting patterns | Day-of-week x hour heatmap | Heatmap |
| Recovery stats | Wayback/PullPush success rates | Stacked bar |

### Intelligent Insights

| Feature | Approach | LLM Required? |
|---------|----------|---------------|
| Interest profile | TF-IDF on titles → weighted by recency | No |
| Topic trends | Monthly topic counts → % change | No |
| Cross-subreddit topics | Keyword overlap between subreddits | No |
| Reading time stats | Word count / 200 wpm | No |
| Saving velocity | Saves per day/week trend | No |
| Topic summaries | LLM summarize top posts per topic | Yes (optional) |
| Interest evolution narrative | LLM describe how interests changed | Yes (optional) |

## Dashboard Framework

### Recommended: Streamlit

- **Why**: Fastest Python-to-web-app path (<100 lines for full dashboard)
- **Built-in**: Charts (Plotly), data tables, widgets, session state
- **Chat UI**: `st.chat_message` and `st.chat_input` for RAG chat
- **Deploy**: Local (`streamlit run app.py`), Streamlit Cloud (free), Docker

### Alternative: Static HTML Report

- **Plotly** charts → `fig.write_html()` (interactive in static HTML)
- **Jinja2** templates for layout
- **Pro**: Host on GitHub Pages, no server needed
- **Con**: Not interactive (no re-sorting, filtering)

## Sentiment Analysis (Lightweight)

### Recommended: VADER (No LLM)

- Rule-based, ~200KB, zero dependencies
- Optimized for social media (handles slang, emojis, ALL CAPS)
- Returns compound score -1 (negative) to +1 (positive)
- 99.5% accuracy on Reddit comments

```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
score = analyzer.polarity_scores("This is absolutely amazing!")
# {'neg': 0.0, 'neu': 0.328, 'pos': 0.672, 'compound': 0.6588}
```

## Post Categorization (Rule-Based, No ML)

```python
def categorize_post(title: str, is_self: bool, has_media: bool) -> str:
    title_lower = title.lower()
    if '?' in title or title_lower.startswith(('how', 'why', 'what', 'is ', 'can ')):
        return 'question'
    if any(w in title_lower for w in ('guide', 'tutorial', 'how to', 'eli5')):
        return 'tutorial'
    if has_media:
        return 'media'
    if is_self:
        return 'discussion'
    return 'link'
```

## Obsidian / PKM Compatibility

Reddit Stash markdown files are already 90% compatible with Obsidian.
To enable graph view, add wikilinks based on similarity:

```yaml
---
id: abc123
subreddit: programming
related: [[POST_xyz789]] [[POST_def456]]
tags: [rust, async, performance]
---
```

Auto-generate `related` links via cosine similarity on embeddings.

## Sources

- Streamlit: https://streamlit.io
- VADER Sentiment: https://github.com/cjhutto/vaderSentiment
- Plotly: https://plotly.com/python/
