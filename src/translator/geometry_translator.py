import re

class GeometryTranslator:
    """
    Advanced Geometry Proof Translator (Master Version).
    Handles AlphaGeometry symbolic tokens, point indices, and rule mapping.
    """
    
    RULE_MAP = {
        "r00": "Perpendiculars give parallel",
        "r01": "Definition of cyclic",
        "r02": "Parallel from inclination",
        "r03": "Arc determines internal angles",
        "r04": "Congruent angles are in a cyclic",
        "r05": "Same arc same chord",
        "r06": "Base of half triangle",
        "r07": "Thales Theorem I",
        "r08": "Right triangles common angle I",
        "r09": "Sum of angles of a triangle",
        "r10": "Ratio cancellation",
        "r11": "Bisector theorem I",
        "r12": "Bisector theorem II",
        "r13": "Isosceles triangle equal angles",
        "r14": "Equal base angles imply isosceles",
        "r15": "Arc determines inscribed angles (tangent)",
        "r16": "Same arc giving tangent",
        "r17": "Central angle vs inscribed angle I",
        "r18": "Central angle vs inscribed angle II",
        "r19": "Hypotenuse is diameter",
        "r20": "Diameter is hypotenuse",
        "r21": "Cyclic trapezoid",
        "r22": "Bisector construction",
        "r23": "Bisector is perpendicular",
        "r24": "Cyclic kite",
        "r25": "Diagonals of parallelogram I",
        "r26": "Diagonals of parallelogram II",
        "r27": "Thales theorem II",
        "r28": "Overlapping parallels",
        "r29": "Midpoint Theorem",
        "r30": "Right triangles common angle II",
        "r31": "Denominator cancelling",
        "r34": "AA Similarity (Direct)",
        "r35": "AA Similarity (Reverse)",
        "r36": "ASA Congruence (Direct)",
        "r37": "ASA Congruence (Reverse)",
        "r41": "Thales theorem III",
        "r42": "Thales theorem IV",
        "r43": "Orthocenter theorem",
        "r44": "Pappus's theorem",
        "r45": "Simson's line theorem",
        "r46": "Incenter theorem",
        "r47": "Circumcenter theorem",
        "r48": "Centroid theorem",
        "r51": "Midpoint splits in two",
        "r54": "Definition of midpoint",
        "r57": "Pythagoras theorem",
        "r72": "Disassembling a circle",
        "r73": "Definition of circle"
    }

    MAP_SYMBOL = {
        "T": "⟂",
        "P": "∥",
        "D": "≅",
        "S": "∼",
        "I": "Circle",
        "M": "Midpoint",
        "O": "Cyclic",
        "C": "Collinear",
        "^": "∠",
        "/": "Ratio",
        "%": "Ratio",
        "=": "≅",
        "perp": "⟂",
        "para": "∥",
        "parallel": "∥",
        "cong": "≅",
        "sim": "∼",
        "angle": "∠",
        "coll": "Collinear",
        "midpoint": "Midpoint",
        "intersection": "Intersection",
        "circle": "Circle",
        "cyclic": "Cyclic"
    }

    def __init__(self):
        self.point_map = {}

    def setup_point_map(self, prompt: str):
        """Extracts points from prompt and maps indices to names."""
        initial_points = re.findall(r'\b([a-z])\b', prompt.split('?')[0])
        self.point_map = {}
        
        # Map 0, 1, 2... and 00, 01, 02... to point names
        for i, name in enumerate(initial_points):
            self.point_map[str(i)] = name.upper()
            self.point_map[f"{i:02d}"] = name.upper()
            self.point_map[name.lower()] = name.upper()
        
        # Add auxiliary points (alphabetical)
        all_alphabet = "abcdefghijklmnopqrstuvwxyz" + "abcdefghijklmnopqrstuvwxyz".upper()
        for i in range(len(initial_points), 100):
            idx_in_alpha = i - len(initial_points)
            if idx_in_alpha < len(all_alphabet):
                name = all_alphabet[idx_in_alpha]
            else:
                name = f"P{i}"
            self.point_map[str(i)] = name.upper()
            self.point_map[f"{i:02d}"] = name.upper()

    def translate_token(self, token: str) -> str:
        """Translates a single token based on POINT_MAP and RULE_MAP."""
        t_clean = re.sub(r'[^a-z0-9;:]', '', token.lower())
        
        if not t_clean:
            return ""

        # Check for rules in the rule map
        if t_clean in self.RULE_MAP:
            return f"[{self.RULE_MAP[t_clean]}]"
        
        # Check for points (indices or names)
        if t_clean in self.point_map:
            return self.point_map[t_clean]

        # Check for symbolic predicates or constructive keywords
        if t_clean in self.MAP_SYMBOL:
            return self.MAP_SYMBOL[t_clean]
        
        upper_t = token.upper()
        if upper_t in self.MAP_SYMBOL:
            return self.MAP_SYMBOL[upper_t]
        
        # Check for generic rules starting with r
        if t_clean.startswith("r") and len(t_clean) > 1 and t_clean[1].isdigit():
            return f"[Rule {t_clean.upper()}]"
        
        # Punctuation
        if t_clean in [";", ":", "?", "="]:
            return t_clean
            
        return t_clean.upper()

    def translate_proof(self, prompt: str, solution_raw: str) -> str:
        """Translates the entire raw solution into a human-readable proof."""
        self.setup_point_map(prompt)
        
        # Pre-process: split merged tokens like '02perp' or 'r29para'
        # This regex looks for digits followed by letters, or 'r' digits followed by letters
        processed_raw = re.sub(r'(\d+)([a-z]+)', r'\1 \2', solution_raw.lower())
        processed_raw = re.sub(r'(r\d+)([a-z]+)', r'\1 \2', processed_raw)
        
        # Clean the raw solution
        clean_solution = re.sub(r'[^a-z0-9\s;:\?=\-\+]', ' ', processed_raw)
        
        # Tokenize (updated to handle more patterns)
        raw_tokens = re.findall(r'r[0-9]+|[a-z]+|[0-9]+|;|:|\?|=', clean_solution)
        
        translated_parts = []
        step_count = 1
        
        for t in raw_tokens:
            translated = self.translate_token(t)
            if not translated:
                continue
            
            # Start new step on rules
            if "[" in translated and "]" in translated:
                translated_parts.append(f"\nSTEP {step_count:02d}: Using {translated} on")
                step_count += 1
            else:
                translated_parts.append(translated)
        
        final_proof = " ".join(translated_parts).replace(" ;", ";").replace(" :", ":").replace(" ?", "?")
        return final_proof.strip()

if __name__ == "__main__":
    # Test
    translator = GeometryTranslator()
    prompt = "a b c = triangle a b c; d = midpoint a b; e = midpoint a c; ? parallel d e b c"
    raw = "a b c = triangle a b c; d = midpoint a b; e = midpoint a c; ? parallel d e b c 02 r29 00 r42 06 ^ 11 02 e 02 e f ;"
    print(translator.translate_proof(prompt, raw))
