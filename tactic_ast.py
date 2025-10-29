"""
AST-based Tactic Representation for ASTactic
Represents tactics as Abstract Syntax Trees
Based on Yang & Deng 2019 - Learning to Prove Theorems via Interacting with Proof Assistants
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from enum import Enum
import json


class TacticNodeType(Enum):
    """Types of nodes in tactic AST"""
    # Terminal nodes
    IDENTIFIER = "identifier"
    NUMBER = "number"
    STRING = "string"
    
    # Tactic nodes
    APPLY = "apply"
    INTRO = "intro"
    INTROS = "intros"
    REWRITE = "rewrite"
    INDUCTION = "induction"
    CASES = "cases"
    REFLEXIVITY = "reflexivity"
    SYMMETRY = "symmetry"
    TRANSITIVITY = "transitivity"
    ASSUMPTION = "assumption"
    EXACT = "exact"
    SPLIT = "split"
    LEFT = "left"
    RIGHT = "right"
    EXISTS = "exists"
    CONSTRUCTOR = "constructor"
    DESTRUCT = "destruct"
    SIMPL = "simpl"
    UNFOLD = "unfold"
    RING = "ring"
    OMEGA = "omega"
    AUTO = "auto"
    
    # Combinators
    SEQUENCE = "sequence"  # tactic1; tactic2
    REPEAT = "repeat"
    TRY = "try"
    
    # Arguments
    ARGUMENT = "argument"
    TERM = "term"


@dataclass
class ASTNode:
    """Node in Abstract Syntax Tree for tactics"""
    node_type: TacticNodeType
    value: Optional[str] = None
    children: List['ASTNode'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'type': self.node_type.value,
            'value': self.value,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASTNode':
        """Create from dictionary"""
        node = cls(
            node_type=TacticNodeType(data['type']),
            value=data.get('value'),
            children=[]
        )
        for child_data in data.get('children', []):
            node.children.append(cls.from_dict(child_data))
        return node
    
    def to_string(self) -> str:
        """Convert AST to tactic string"""
        if self.node_type == TacticNodeType.IDENTIFIER:
            return self.value
        
        if self.node_type == TacticNodeType.NUMBER:
            return str(self.value)
        
        if self.node_type == TacticNodeType.STRING:
            return f'"{self.value}"'
        
        # Simple tactics
        if self.node_type in [
            TacticNodeType.REFLEXIVITY, 
            TacticNodeType.SYMMETRY,
            TacticNodeType.ASSUMPTION,
            TacticNodeType.AUTO,
            TacticNodeType.SIMPL
        ]:
            return self.node_type.value
        
        # Tactics with arguments
        if self.node_type == TacticNodeType.APPLY:
            args = ' '.join(child.to_string() for child in self.children)
            return f"apply {args}"
        
        if self.node_type == TacticNodeType.INTRO:
            if self.children:
                args = ' '.join(child.to_string() for child in self.children)
                return f"intro {args}"
            return "intro"
        
        if self.node_type == TacticNodeType.INTROS:
            if self.children:
                args = ' '.join(child.to_string() for child in self.children)
                return f"intros {args}"
            return "intros"
        
        if self.node_type == TacticNodeType.REWRITE:
            args = ' '.join(child.to_string() for child in self.children)
            return f"rewrite {args}"
        
        if self.node_type == TacticNodeType.INDUCTION:
            args = ' '.join(child.to_string() for child in self.children)
            return f"induction {args}"
        
        if self.node_type == TacticNodeType.CASES:
            args = ' '.join(child.to_string() for child in self.children)
            return f"cases {args}"
        
        if self.node_type == TacticNodeType.DESTRUCT:
            args = ' '.join(child.to_string() for child in self.children)
            return f"destruct {args}"
        
        if self.node_type == TacticNodeType.EXACT:
            args = ' '.join(child.to_string() for child in self.children)
            return f"exact {args}"
        
        if self.node_type == TacticNodeType.EXISTS:
            args = ' '.join(child.to_string() for child in self.children)
            return f"exists {args}"
        
        if self.node_type == TacticNodeType.UNFOLD:
            args = ' '.join(child.to_string() for child in self.children)
            return f"unfold {args}"
        
        # Combinators
        if self.node_type == TacticNodeType.SEQUENCE:
            tactics = '; '.join(child.to_string() for child in self.children)
            return tactics
        
        if self.node_type == TacticNodeType.REPEAT:
            tactic = self.children[0].to_string() if self.children else ""
            return f"repeat ({tactic})"
        
        if self.node_type == TacticNodeType.TRY:
            tactic = self.children[0].to_string() if self.children else ""
            return f"try ({tactic})"
        
        return f"<{self.node_type.value}>"
    
    def size(self) -> int:
        """Return size of AST (number of nodes)"""
        return 1 + sum(child.size() for child in self.children)
    
    def depth(self) -> int:
        """Return depth of AST"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)


class TacticParser:
    """Parse natural language tactics into AST"""
    
    def __init__(self):
        # Mapping from natural language patterns to tactic types
        self.patterns = {
            'apply': TacticNodeType.APPLY,
            'use': TacticNodeType.APPLY,
            'introduce': TacticNodeType.INTRO,
            'let': TacticNodeType.INTRO,
            'assume': TacticNodeType.INTRO,
            'rewrite': TacticNodeType.REWRITE,
            'substitute': TacticNodeType.REWRITE,
            'induction on': TacticNodeType.INDUCTION,
            'induct': TacticNodeType.INDUCTION,
            'case analysis': TacticNodeType.CASES,
            'consider cases': TacticNodeType.CASES,
            'by reflexivity': TacticNodeType.REFLEXIVITY,
            'trivial': TacticNodeType.REFLEXIVITY,
            'obvious': TacticNodeType.REFLEXIVITY,
            'by assumption': TacticNodeType.ASSUMPTION,
            'from hypothesis': TacticNodeType.ASSUMPTION,
            'exactly': TacticNodeType.EXACT,
            'we have': TacticNodeType.EXACT,
            'split': TacticNodeType.SPLIT,
            'simplify': TacticNodeType.SIMPL,
            'unfold': TacticNodeType.UNFOLD,
            'exists': TacticNodeType.EXISTS,
            'choose': TacticNodeType.EXISTS,
        }
    
    def parse(self, text: str) -> ASTNode:
        """
        Parse natural language proof step into tactic AST
        
        This is a simplified parser. A real implementation would need
        more sophisticated NLP and pattern matching.
        """
        text = text.lower().strip()
        
        # Try to match patterns
        for pattern, tactic_type in self.patterns.items():
            if pattern in text:
                return self._parse_tactic(text, pattern, tactic_type)
        
        # Default: treat as assumption
        return ASTNode(
            node_type=TacticNodeType.ASSUMPTION,
            value=text
        )
    
    def _parse_tactic(
        self, 
        text: str, 
        pattern: str, 
        tactic_type: TacticNodeType
    ) -> ASTNode:
        """Parse specific tactic with arguments"""
        
        # Extract argument after pattern
        idx = text.find(pattern)
        if idx >= 0:
            arg_text = text[idx + len(pattern):].strip()
            
            # Remove common words
            arg_text = arg_text.replace('that', '').replace('the', '').strip()
            
            if arg_text:
                # Create argument node
                arg_node = ASTNode(
                    node_type=TacticNodeType.IDENTIFIER,
                    value=arg_text
                )
                
                return ASTNode(
                    node_type=tactic_type,
                    children=[arg_node]
                )
        
        return ASTNode(node_type=tactic_type)
    
    def parse_proof_steps(self, steps: List[str]) -> List[ASTNode]:
        """Parse list of proof steps into AST list"""
        return [self.parse(step) for step in steps]


class TacticGenerator:
    """Generate tactics from AST"""
    
    @staticmethod
    def generate_lean(ast: ASTNode) -> str:
        """Generate Lean tactic from AST"""
        
        # Map to Lean tactic syntax
        lean_mapping = {
            TacticNodeType.APPLY: 'apply',
            TacticNodeType.INTRO: 'intro',
            TacticNodeType.INTROS: 'intros',
            TacticNodeType.REWRITE: 'rw',
            TacticNodeType.INDUCTION: 'induction',
            TacticNodeType.CASES: 'cases',
            TacticNodeType.REFLEXIVITY: 'rfl',
            TacticNodeType.SYMMETRY: 'symm',
            TacticNodeType.ASSUMPTION: 'assumption',
            TacticNodeType.EXACT: 'exact',
            TacticNodeType.SPLIT: 'split',
            TacticNodeType.SIMPL: 'simp',
            TacticNodeType.UNFOLD: 'unfold',
            TacticNodeType.EXISTS: 'use',
            TacticNodeType.AUTO: 'simp',
        }
        
        if ast.node_type in lean_mapping:
            lean_tactic = lean_mapping[ast.node_type]
            if ast.children:
                args = ' '.join(child.to_string() for child in ast.children)
                return f"{lean_tactic} {args}"
            return lean_tactic
        
        return ast.to_string()
    
    @staticmethod
    def generate_coq(ast: ASTNode) -> str:
        """Generate Coq tactic from AST"""
        # For Coq, the AST string representation is already close
        return ast.to_string()


# Vocabulary for AST nodes
class TacticVocabulary:
    """Vocabulary for encoding/decoding tactics"""
    
    def __init__(self):
        self.node_type_to_idx = {
            node_type: idx 
            for idx, node_type in enumerate(TacticNodeType)
        }
        self.idx_to_node_type = {
            idx: node_type 
            for node_type, idx in self.node_type_to_idx.items()
        }
        
        # Token vocabulary for identifiers
        self.token_to_idx = {'<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.next_idx = 4
    
    def add_token(self, token: str):
        """Add token to vocabulary"""
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.next_idx
            self.idx_to_token[self.next_idx] = token
            self.next_idx += 1
    
    def encode_node_type(self, node_type: TacticNodeType) -> int:
        """Encode node type to index"""
        return self.node_type_to_idx[node_type]
    
    def decode_node_type(self, idx: int) -> TacticNodeType:
        """Decode index to node type"""
        return self.idx_to_node_type[idx]
    
    def encode_token(self, token: str) -> int:
        """Encode token to index"""
        return self.token_to_idx.get(token, self.token_to_idx['<UNK>'])
    
    def decode_token(self, idx: int) -> str:
        """Decode index to token"""
        return self.idx_to_token.get(idx, '<UNK>')
    
    def size(self) -> int:
        """Return vocabulary size"""
        return len(self.token_to_idx)


if __name__ == "__main__":
    # Test AST representation
    print("=== Testing Tactic AST ===\n")
    
    # Create simple tactic: apply theorem_name
    tactic1 = ASTNode(
        node_type=TacticNodeType.APPLY,
        children=[
            ASTNode(node_type=TacticNodeType.IDENTIFIER, value="le_refl")
        ]
    )
    print(f"Tactic 1: {tactic1.to_string()}")
    print(f"Size: {tactic1.size()}, Depth: {tactic1.depth()}\n")
    
    # Create complex tactic: intro x; induction x
    tactic2 = ASTNode(
        node_type=TacticNodeType.SEQUENCE,
        children=[
            ASTNode(
                node_type=TacticNodeType.INTRO,
                children=[ASTNode(node_type=TacticNodeType.IDENTIFIER, value="x")]
            ),
            ASTNode(
                node_type=TacticNodeType.INDUCTION,
                children=[ASTNode(node_type=TacticNodeType.IDENTIFIER, value="x")]
            )
        ]
    )
    print(f"Tactic 2: {tactic2.to_string()}")
    print(f"Size: {tactic2.size()}, Depth: {tactic2.depth()}\n")
    
    # Test parser
    print("=== Testing Parser ===\n")
    parser = TacticParser()
    
    natural_steps = [
        "Apply the theorem le_refl",
        "Introduce variable x",
        "Perform induction on x",
        "By reflexivity",
        "Use assumption"
    ]
    
    for step in natural_steps:
        ast = parser.parse(step)
        print(f"Input: {step}")
        print(f"AST: {ast.to_string()}")
        print(f"Lean: {TacticGenerator.generate_lean(ast)}")
        print()
    
    # Test serialization
    print("=== Testing Serialization ===\n")
    ast_dict = tactic2.to_dict()
    print(f"Dict: {json.dumps(ast_dict, indent=2)}\n")
    
    reconstructed = ASTNode.from_dict(ast_dict)
    print(f"Reconstructed: {reconstructed.to_string()}")
    print(f"Match: {reconstructed.to_string() == tactic2.to_string()}")
