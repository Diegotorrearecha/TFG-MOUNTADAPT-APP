from fpdf import FPDF
import datetime

# Función para limpiar caracteres no compatibles con latin1
def clean_text(text):
    replacements = {
        "–": "-",  # guion largo
        "—": "-",  # guion em
        "“": '"', "”": '"',  # comillas curvas
        "‘": "'", "’": "'",  # apóstrofes
        "•": "-", "→": "->", "°": " degrees"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

# Clase PDF personalizada
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "MountAdapt - Project Summary", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, clean_text(title), ln=True)
        self.set_font("Arial", "", 11)
        self.multi_cell(0, 8, clean_text(content))
        self.ln(5)

# Función principal para generar el PDF
def create_pdf(summary_data: dict) -> bytes:
    pdf = PDFReport()
    pdf.add_page()

    for title, content in summary_data.items():
        pdf.add_section(title, content)

    return pdf.output(dest='S').encode('latin1')
