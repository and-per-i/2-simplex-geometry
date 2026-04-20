import re

class GeometryTranslator:
    """
    Translates geometric proofs from a symbolic/specific language to human-readable format.
    Inspired by Newclid and AlphaGeometry.
    """
    
    MAP_SYMBOL = {
        "T": "perp",
        "P": "para",
        "D": "cong",
        "S": "simtri",
        "I": "circle",
        "M": "midp",
        "O": "cyclic",
        "C": "coll",
        "^": "eqangle",
        "/": "eqratio",
        "%": "eqratio",
        "=": "contri",
        "X": "collx",
        "A": "acompute",
        "R": "rcompute",
        "Q": "fixc",
        "E": "fixl",
        "V": "fixb",
        "H": "fixt",
        "Z": "fixp",
        "Y": "ind",
    }

    PREDICATE_SYMBOLS = {
        "perp": "⟂",
        "para": "∥",
        "cong": "≅",
        "cyclic": "is cyclic",
        "coll": "are collinear",
        "midp": "is midpoint of",
        "simtri": "∼",
        "contri": "≅",
        "eqangle": "∠",
        "eqratio": "ratio",
    }

    RULE_DESCRIPTIONS = {
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
        "r29": "Midpoint is an eqratio",
        "r30": "Right triangles common angle II",
        "r31": "Denominator cancelling",
        "r34": "AA Similarity of triangles (Direct)",
        "r35": "AA Similarity of triangles (Reverse)",
        "r36": "ASA Congruence of triangles (Direct)",
        "r37": "ASA Congruence of triangles (Reverse)",
        "r41": "Thales theorem III",
        "r42": "Thales theorem IV",
        "r43": "Orthocenter theorem",
        "r44": "Pappus's theorem",
        "r45": "Simson's line theorem",
        "r46": "Incenter theorem",
        "r47": "Circumcenter theorem",
        "r48": "Centroid theorem",
    }

    def translate_predicate(self, predicate_str: str) -> str:
        """Translates a symbolic predicate to a human-readable string."""
        parts = predicate_str.split()
        if not parts:
            return ""
        
        name = parts[0]
        args = parts[1:]
        
        # Normalize name using MAP_SYMBOL
        if name in self.MAP_SYMBOL:
            name = self.MAP_SYMBOL[name]
        
        if name == "perp":
            return f"{args[0]}{args[1]} ⟂ {args[2]}{args[3]}"
        elif name == "para":
            return f"{args[0]}{args[1]} ∥ {args[2]}{args[3]}"
        elif name == "cong":
            return f"{args[0]}{args[1]} ≅ {args[2]}{args[3]}"
        elif name == "coll":
            return f"{', '.join(args)} are collinear"
        elif name == "cyclic":
            return f"{', '.join(args)} are concyclic"
        elif name == "midp":
            return f"{args[0]} is midpoint of {args[1]}{args[2]}"
        elif name == "eqangle":
            # eqangle A B C D E F G H -> ∠(AB, CD) = ∠(EF, GH)
            if len(args) == 8:
                return f"∠({args[0]}{args[1]}, {args[2]}{args[3]}) = ∠({args[4]}{args[5]}, {args[6]}{args[7]})"
            return f"∠({' '.join(args)})"
        elif name == "eqratio":
            # eqratio A B C D E F G H -> AB/CD = EF/GH
            if len(args) == 8:
                return f"{args[0]}{args[1]}/{args[2]}{args[3]} = {args[4]}{args[5]}/{args[6]}{args[7]}"
            return f"ratio({' '.join(args)})"
        elif name == "simtri":
            if len(args) == 6:
                return f"△{args[0]}{args[1]}{args[2]} ∼ △{args[3]}{args[4]}{args[5]}"
            return f"simtri({' '.join(args)})"
        elif name == "contri":
            if len(args) == 6:
                return f"△{args[0]}{args[1]}{args[2]} ≅ △{args[3]}{args[4]}{args[5]}"
            return f"contri({' '.join(args)})"
        
        return f"{name}({', '.join(args)})"

    def translate_step(self, step_id: int, premises: list[str], rule_id: str, conclusion: str) -> str:
        """Translates a single proof step."""
        human_premises = " & ".join([self.translate_predicate(p) for p in premises])
        rule_desc = self.RULE_DESCRIPTIONS.get(rule_id, rule_id)
        human_conclusion = self.translate_predicate(conclusion)
        
        return f"{step_id:03d}. {human_premises} ⇒ ({rule_desc}) ⇒ {human_conclusion}"

    def translate_constrained_to_constructive(self, point: str, name: str, args: list[str]) -> tuple[str, list[str]]:
        """
        Translates a predicate from constraint-based to construction-based.
        Useful for interpreting LM outputs.
        """
        # Normalize name
        if name in self.MAP_SYMBOL:
            name = self.MAP_SYMBOL[name]

        if name in ["perp", "T"]:
            a, b, c, d = args
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            if point == d:
                c, d = d, c
            if a == c and a == point:
                return "on_dia", [a, b, d]
            return "on_tline", [a, b, c, d]

        elif name in ["para", "P"]:
            a, b, c, d = args
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            return "on_pline", [a, b, c, d]

        elif name in ["cong", "D"]:
            a, b, c, d = args
            if point in [c, d]:
                a, b, c, d = c, d, a, b
            if point == b:
                a, b = b, a
            if point == d:
                c, d = d, c
            if a == c and a == point:
                return "on_bline", [a, b, d]
            if b in [c, d]:
                if b == d:
                    c, d = d, c
                return "on_circle", [a, b, d]
            return "eqdistance", [a, b, c, d]

        elif name in ["coll", "C"]:
            a, b, c = args
            if point == b:
                a, b = b, a
            if point == c:
                a, b, c = c, a, b
            return "on_line", [a, b, c]

        elif name in ["cyclic", "O"]:
            a, b, c = [x for x in args if x != point]
            return "on_circum", [point, a, b, c]

        return name, args

    def translate_proof(self, proof_steps: list[dict]) -> str:
        """
        Translates a list of proof steps.
        Each step should be a dict with: 'premises', 'rule', 'conclusion'.
        """
        lines = []
        for i, step in enumerate(proof_steps):
            lines.append(self.translate_step(i + 1, step['premises'], step['rule'], step['conclusion']))
        return "\n".join(lines)

if __name__ == "__main__":
    translator = GeometryTranslator()
    
    # Example steps
    steps = [
        {
            "premises": ["T a b c d", "T c d e f"],
            "rule": "r00",
            "conclusion": "P a b e f"
        },
        {
            "premises": ["O a b c d"],
            "rule": "r03",
            "conclusion": "^ a c a d b c b d"
        }
    ]
    
    print("Proof Translation Example:")
    print(translator.translate_proof(steps))
