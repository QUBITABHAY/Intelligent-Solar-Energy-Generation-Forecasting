"""
PDF Report Generator for the Solar Grid Optimization Assistant.

Generates a structured, downloadable PDF report containing:
1. Solar generation forecast summary
2. Identified variability and risk periods
3. Grid balancing and storage recommendations
4. Energy utilization optimization strategies
5. Supporting references
"""

from fpdf import FPDF
from datetime import datetime


class SolarReportPDF(FPDF):
    """Custom PDF class with header/footer for solar reports."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Solar Grid Optimization Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(1)
        self.set_draw_color(0, 120, 200)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        """Render a styled section heading."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(0, 80, 160)
        # Encode to latin-1 safe text (same as section_body)
        safe_title = title.encode("latin-1", errors="replace").decode("latin-1")
        self.cell(0, 10, safe_title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 120, 200)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_body(self, text: str):
        """Render section body text with word wrapping."""
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        # Encode to latin-1 safe text (fpdf2 default)
        safe_text = text.encode("latin-1", errors="replace").decode("latin-1")
        self.multi_cell(0, 5.5, safe_text)
        self.ln(4)

    def key_value_row(self, key: str, value: str):
        """Render a bold key with a normal value on one line."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(40, 40, 40)
        self.cell(55, 6, f"{key}:")
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")


def generate_report(
    recommendation: str,
    forecast_summary: dict | None = None,
    risks: list | None = None,
) -> bytes:
    """
    Generate a PDF report from the agent's output.

    Args:
        recommendation: The full structured text from the LLM (Groq/Gemini).
        forecast_summary: Dict from analyze_forecast() with mean, max, min, std.
        risks: List of risk period dicts from identify_risks().

    Returns:
        PDF file content as bytes (ready for st.download_button).
    """
    pdf = SolarReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(0, 60, 130)
    pdf.cell(0, 12, "Solar Grid Optimization Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(8)

    # ── Section 1: Forecast Summary (from data) ──
    if forecast_summary:
        pdf.section_title("1. Solar Generation Forecast Summary")

        def _fmt(val) -> str:
            """Safely format a numeric value, returning 'N/A' for missing keys."""
            try:
                return f"{float(val):.2f} kW"
            except (TypeError, ValueError):
                return "N/A"

        pdf.key_value_row("Mean Generation", _fmt(forecast_summary.get('mean')))
        pdf.key_value_row("Max Generation", _fmt(forecast_summary.get('max')))
        pdf.key_value_row("Min Generation", _fmt(forecast_summary.get('min')))
        pdf.key_value_row("Std Deviation", _fmt(forecast_summary.get('std')))

        peak_count = len(forecast_summary.get("peak_indices", []))
        low_count = len(forecast_summary.get("low_indices", []))
        pdf.key_value_row("Peak Generation Periods", f"{peak_count} time steps")
        pdf.key_value_row("Low Generation Periods", f"{low_count} time steps")
        pdf.ln(4)

    # ── Section 2: Risk Periods (from data) ──
    if risks is not None:
        pdf.section_title("2. Identified Variability & Risk Periods")
        if len(risks) == 0:
            pdf.section_body("No significant risk periods were identified in the forecast data.")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(40, 40, 40)
            pdf.cell(0, 6, f"Total risk periods detected: {len(risks)}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)

            # Risk summary table
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(0, 80, 160)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(30, 7, "Period", border=1, fill=True, align="C")
            pdf.cell(35, 7, "Start Index", border=1, fill=True, align="C")
            pdf.cell(35, 7, "End Index", border=1, fill=True, align="C")
            pdf.cell(40, 7, "Avg Power (kW)", border=1, fill=True, align="C")
            pdf.cell(30, 7, "Risk Level", border=1, fill=True, align="C")
            pdf.ln()

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(40, 40, 40)
            for i, risk in enumerate(risks[:20], 1):  # Show top 20
                # Color-code risk level
                level = risk.get("risk_level", "MEDIUM")
                if level == "HIGH":
                    pdf.set_text_color(200, 0, 0)
                elif level == "MEDIUM":
                    pdf.set_text_color(200, 130, 0)
                else:
                    pdf.set_text_color(40, 40, 40)

                pdf.cell(30, 6, str(i), border=1, align="C")
                pdf.cell(35, 6, str(risk.get("start_index", "")), border=1, align="C")
                pdf.cell(35, 6, str(risk.get("end_index", "")), border=1, align="C")
                pdf.cell(40, 6, f"{risk.get('avg_power', 0):.1f}", border=1, align="C")
                pdf.cell(30, 6, level, border=1, align="C")
                pdf.ln()
                pdf.set_text_color(40, 40, 40)

            if len(risks) > 20:
                pdf.ln(2)
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(0, 6, f"... and {len(risks) - 20} more risk periods.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

    # ── Sections 3-5: LLM Recommendation ──
    if recommendation:
        # Try to split the recommendation into sections if the LLM used markdown headers
        sections = _split_recommendation(recommendation)

        if sections:
            for title, body in sections:
                pdf.section_title(title)
                pdf.section_body(body)
        else:
            # Fallback: just dump the whole recommendation
            pdf.section_title("3. Grid Optimization Recommendations")
            pdf.section_body(recommendation)

    return bytes(pdf.output())


def _split_recommendation(text: str) -> list[tuple[str, str]]:
    """
    Attempt to split LLM output into titled sections.
    Looks for markdown-style headers (## or **Title**).
    Returns list of (title, body) tuples. Empty list if no structure found.
    """
    import re

    # Try markdown ## headers first (match at start of string or after newline)
    parts = re.split(r'(?:^|\n)#{1,3}\s+', text)
    if len(parts) > 2:
        sections = []
        for part in parts[1:]:  # Skip text before first header
            lines = part.strip().split('\n', 1)
            title = lines[0].strip().rstrip('#').strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if title:
                sections.append((title, body))
        return sections

    # Try **bold** headers
    parts = re.split(r'(?:^|\n)\*\*(.+?)\*\*\s*\n', text)
    if len(parts) > 2:
        sections = []
        for i in range(1, len(parts), 2):
            title = parts[i].strip()
            body = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if title:
                sections.append((title, body))
        return sections

    # Try numbered sections like "1. Title" or "1) Title" 
    parts = re.split(r'(?:^|\n)\d+[\.\)]\s+', text)
    if len(parts) > 2:
        # Re-extract with titles
        matches = list(re.finditer(r'(?:^|\n)(\d+[\.\)]\s+.+)', text))
        if matches:
            sections = []
            for j, match in enumerate(matches):
                title = match.group(1).strip()
                start = match.end()
                end = matches[j + 1].start() if j + 1 < len(matches) else len(text)
                body = text[start:end].strip()
                sections.append((title, body))
            return sections

    return []


if __name__ == "__main__":
    # Quick test
    test_summary = {
        "mean": 1234.56,
        "max": 2800.00,
        "min": 0.00,
        "std": 890.12,
        "peak_indices": list(range(50)),
        "low_indices": list(range(30)),
    }

    test_risks = [
        {"start_index": 10, "end_index": 15, "avg_power": 120.5, "risk_level": "HIGH"},
        {"start_index": 45, "end_index": 52, "avg_power": 380.2, "risk_level": "MEDIUM"},
    ]

    test_recommendation = """
## 1. Solar Generation Forecast Summary
The solar plant shows strong midday generation with an average output of 1234 kW.

## 2. Identified Variability and Risk Periods
Two critical periods were identified where generation dropped below safe thresholds.

## 3. Grid Balancing and Storage Recommendations
Deploy battery storage of 300 kWh capacity to cover evening peak demand shortfall.

## 4. Energy Utilization Optimization Strategies
Shift heavy industrial loads to the 10 AM - 2 PM window to maximize solar self-consumption.

## 5. Supporting References
Based on grid management guidelines and battery storage best practices from the knowledge base.
"""

    pdf_bytes = generate_report(test_recommendation, test_summary, test_risks)

    with open("test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"Test report generated: test_report.pdf ({len(pdf_bytes)} bytes)")
