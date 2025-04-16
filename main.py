import tkinter as tk
import os
import datetime
import sys
import subprocess
import random
from model import RhymingTextGenerator
import torch

class PeppenwriterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("peppenwriter")
        
        # Vollbildmodus aktivieren
        self.root.attributes('-fullscreen', True)
        
        # Hintergrund komplett schwarz
        self.root.configure(bg='black')
        
        # Zeilenzähler
        self.line_count = 0
        self.max_lines = 49  # Maximal 49 Zeilen
        self.min_lines = 4   # Mindestens 4 Zeilen
        
        # Zufällige Zeilenanzahl zwischen min und max für diesen Lauf bestimmen
        self.target_lines = random.randint(self.min_lines, self.max_lines)
        
        # Initialisiere das KI-Modell
        self.model = RhymingTextGenerator()
        
        # Textfeld mit schwarzem Hintergrund und weißer Schrift
        self.text = tk.Text(
            self.root, 
            bg='#000000',           # Schwarzer Hintergrund
            fg='gold',              # Goldene Schrift
            insertbackground='magenta', # Magenta Cursor
            font=('Consolas', 14),  # Schriftart und -größe
            relief='flat',          # Flaches Design
            borderwidth=0,          # Keine Ränder
            highlightthickness=0,   # Keine Rahmen
            padx=20,                # Innenabstand horizontal
            pady=20,                # Innenabstand vertikal
            wrap='word'             # Zeilenumbruch bei Wörtern
        )
        
        # Textfeld über den ganzen Bildschirm
        self.text.pack(fill=tk.BOTH, expand=True)
        
        # Tastenbindungen
        self.text.bind('<Return>', self.on_enter)
        self.text.bind('<Control-Return>', self.on_ctrl_enter)
        
        # ESC-Taste zum Beenden
        self.root.bind('<Escape>', self.save_and_exit)
        
        # Fokus auf das Textfeld setzen
        self.text.focus_set()
        
        # Zeige Starthinweis
        self.show_welcome_message()
    
    def show_welcome_message(self):
        """Zeigt eine Willkommensnachricht an."""
        welcome_text = "peppenwriter_v1\n\nSchreibe eine Zeile und drücke ENTER, um zu beginnen.\nDrücke STRG+ENTER für eine neue Zeile ohne KI-Antwort.\nDrücke ESC zum Speichern und Beenden.\nhappy insanity\n\n"
        self.text.insert(tk.END, welcome_text)
        self.text.tag_add("welcome", "1.0", "end")
        self.text.tag_config("welcome", foreground="#888888")
        
        # Lösche die Willkommensnachricht beim ersten Tastendruck
        def clear_welcome(event):
            self.text.delete("1.0", tk.END)
            self.text.unbind('<Key>', clear_binding)
        
        clear_binding = self.text.bind('<Key>', clear_welcome)
    
    def on_enter(self, event):
        """Verarbeitet das Drücken der Enter-Taste."""
        # Hole die aktuelle Zeile
        current_line = self.get_current_line()
        
        if not current_line.strip():
            return "break"
        
        # Get input length
        input_length = len(current_line.split())
        
        # Zähle die Zeile
        self.line_count += 1
        self.text.insert(tk.END, "\n")
        
        # Generate response with similar length
        ai_response = self.model.generate_response(current_line, target_length=input_length)
        
        self.text.insert(tk.END, ai_response)
        self.line_count += 1
        self.text.insert(tk.END, "\n")
        self.text.see(tk.END)
        
        if self.line_count >= self.min_lines and self.line_count >= self.target_lines:
            self.save_and_exit(None)
        
        return "break"
    
    def on_ctrl_enter(self, event):
        """Verarbeitet das Drücken von Strg+Enter."""
        # Füge einfach einen Zeilenumbruch ein, ohne KI-Antwort
        self.text.insert(tk.END, "\n")
        
        # Zähle die Zeile
        self.line_count += 1
        
        # Prüfe, ob die maximale Zeilenanzahl erreicht ist
        if self.line_count >= self.min_lines and self.line_count >= self.target_lines:
            self.save_and_exit(None)
        
        return "break"
    
    def get_current_line(self):
        """Holt die aktuelle Zeile aus dem Textfeld."""
        # Hole die Position des Cursors
        cursor_pos = self.text.index(tk.INSERT)
        line_num = cursor_pos.split('.')[0]
        
        # Hole den Inhalt der aktuellen Zeile
        line_start = f"{line_num}.0"
        line_end = f"{line_num}.end"
        return self.text.get(line_start, line_end)
    
    def save_and_exit(self, event):
        """Speichert den Text und beendet das Programm."""
        try:
            # Hole den gesamten Text
            full_text = self.text.get("1.0", tk.END)
            
            # Erstelle einen Dateinamen mit Datum und Uhrzeit
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"peppenwriter_{timestamp}.txt"
            
            # Pfad zum Desktop
            desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            file_path = os.path.join(desktop_path, filename)
            
            # Speichere den Text, auch wenn er leer ist
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(full_text)
                
            print(f"Text saved to: {file_path}")
            
        except Exception as e:
            # Bei einem Fehler, versuche in das aktuelle Verzeichnis zu speichern
            try:
                backup_path = f"peppenwriter_backup_{timestamp}.txt"
                with open(backup_path, "w", encoding="utf-8") as file:
                    file.write(full_text)
                print(f"Emergency backup saved to: {backup_path}")
            except Exception as backup_error:
                print(f"Failed to save backup: {str(backup_error)}")
                
        finally:
            # Beende das Programm
            self.root.destroy()

def main():
    # Prüfe CUDA-Verfügbarkeit
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"CUDA ist verfügbar. Gefundene Geräte: {torch.cuda.device_count()}")
        print(f"Aktives Gerät: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA ist nicht verfügbar. Verwende CPU.")
    
    # Starte die GUI
    root = tk.Tk()
    app = PeppenwriterGUI(root)
    root.mainloop()

if __name__ == "__main__":
    # Hide console window if on Windows
    if sys.platform.startswith('win'):
        # If this script is run directly
        if sys.executable.endswith('python.exe'):
            # Re-run with pythonw.exe (hidden console)
            pythonw = sys.executable.replace('python.exe', 'pythonw.exe')
            subprocess.Popen([pythonw] + sys.argv)
            sys.exit()
    
    main()
