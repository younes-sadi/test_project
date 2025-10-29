import torch
8import torch.nn as nn
9import torch.nn.functional as F
10from typing import List, Dict, Tuple, Optional
11from tactic_ast import ASTNode, TacticNodeType, TacticVocabulary
12
13
14class TreeLSTMCell(nn.Module):
15    """
16    Tree-LSTM cell for processing tree structures
17    Used in ASTactic for encoding proof context
18    """
19    
20    def __init__(self, input_size: int, hidden_size: int):
21        super().__init__()
22        self.input_size = input_size
23        self.hidden_size = hidden_size
24        
25        # Gates: input, forget (for each child), output, cell
26        self.W_i = nn.Linear(input_size, hidden_size)
27        self.U_i = nn.Linear(hidden_size, hidden_size)
28        
29        self.W_f = nn.Linear(input_size, hidden_size)
30        self.U_f = nn.Linear(hidden_size, hidden_size)
31        
32        self.W_o = nn.Linear(input_size, hidden_size)
33        self.U_o = nn.Linear(hidden_size, hidden_size)
34        
35        self.W_c = nn.Linear(input_size, hidden_size)
36        self.U_c = nn.Linear(hidden_size, hidden_size)
37    
38    def forward(
39        self, 
40        x: torch.Tensor, 
41        child_h: List[torch.Tensor], 
42        child_c: List[torch.Tensor]
43    ) -> Tuple[torch.Tensor, torch.Tensor]:
44        """
45        Forward pass through Tree-LSTM cell
46        
47        Args:
48            x: Input tensor [batch_size, input_size]
49            child_h: List of child hidden states
50            child_c: List of child cell states
51        
52        Returns:
53            h: Hidden state
54            c: Cell state
55        """
56        
57        # Sum of child hidden states
58        if child_h:
59            h_sum = sum(child_h)
60        else:
61            h_sum = torch.zeros(x.size(0), self.hidden_size, device=x.device)
62        
63        # Input gate
64        i = torch.sigmoid(self.W_i(x) + self.U_i(h_sum))
65        
66        # Output gate
67        o = torch.sigmoid(self.W_o(x) + self.U_o(h_sum))
68        
69        # Cell candidate
70        c_tilde = torch.tanh(self.W_c(x) + self.U_c(h_sum))
71        
72        # Forget gates (one per child)
73        if child_c:
74            f_list = []
75            c_sum = torch.zeros_like(c_tilde)
76            for h_child, c_child in zip(child_h, child_c):
77                f = torch.sigmoid(self.W_f(x) + self.U_f(h_child))
78                f_list.append(f)
79                c_sum = c_sum + f * c_child
80        else:
81            c_sum = torch.zeros_like(c_tilde)
82        
83        # Cell state
84        c = i * c_tilde + c_sum
85        
86        # Hidden state
87        h = o * torch.tanh(c)
88        
89        return h, c
90
91
92class ContextEncoder(nn.Module):
93    """
94    Encodes proof context (goal, hypotheses) into vector representation
95    Uses Tree-LSTM for structured context
96    """
97    
98    def __init__(
99        self, 
100        vocab_size: int,
101        embed_size: int = 256,
102        hidden_size: int = 512,
103        num_layers: int = 2
104    ):
105        super().__init__()
106        
107        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
108        self.encoder = nn.LSTM(
109            embed_size, 
110            hidden_size, 
111            num_layers=num_layers,
112            batch_first=True,
113            bidirectional=True
114        )
115        self.hidden_size = hidden_size
116    
117    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
118        """
119        Encode context into fixed-size representation
120        
121        Args:
122            context_tokens: [batch_size, seq_len]
123        
124        Returns:
125            context_vector: [batch_size, hidden_size * 2]
126        """
127        # Embed tokens
128        embedded = self.embedding(context_tokens)  # [batch, seq_len, embed_size]
129        
130        # Encode with BiLSTM
131        output, (h_n, c_n) = self.encoder(embedded)
132        
133        # Concatenate final forward and backward hidden states
134        # h_n shape: [num_layers * 2, batch, hidden_size]
135        forward_hidden = h_n[-2, :, :]  # Last forward layer
136        backward_hidden = h_n[-1, :, :]  # Last backward layer
137        
138        context_vector = torch.cat([forward_hidden, backward_hidden], dim=1)
139        
140        return context_vector
141
142
143class ASTDecoder(nn.Module):
144    """
145    Generates tactic AST node by node using tree-structured decoder
146    This is the core of ASTactic
147    """
148    
149    def __init__(
150        self,
151        num_node_types: int,
152        vocab_size: int,
153        context_size: int,
154        hidden_size: int = 512,
155        embed_size: int = 256
156    ):
157        super().__init__()
158        
159        self.num_node_types = num_node_types
160        self.vocab_size = vocab_size
161        self.hidden_size = hidden_size
162        
163        # Embeddings
164        self.node_type_embedding = nn.Embedding(num_node_types, embed_size)
165        self.token_embedding = nn.Embedding(vocab_size, embed_size)
166        
167        # LSTM for sequential generation
168        self.lstm = nn.LSTMCell(
169            embed_size + context_size,
170            hidden_size
171        )
172        
173        # Output layers
174        self.node_type_output = nn.Linear(hidden_size, num_node_types)
175        self.token_output = nn.Linear(hidden_size, vocab_size)
176        
177        # Predict number of children
178        self.num_children_output = nn.Linear(hidden_size, 5)  # Max 4 children + 0
179        
180        # Predict whether to continue or stop
181        self.stop_output = nn.Linear(hidden_size, 2)
182    
183    def forward_step(
184        self,
185        prev_embedding: torch.Tensor,
186        context: torch.Tensor,
187        h_state: torch.Tensor,
188        c_state: torch.Tensor
189    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple]:
190        """
191        Single decoding step
192        
193        Returns:
194            node_type_logits: [batch, num_node_types]
195            token_logits: [batch, vocab_size]
196            num_children_logits: [batch, 5]
197            stop_logits: [batch, 2]
198            (h_state, c_state): Updated states
199        """
200        
201        # Concatenate previous embedding and context
202        lstm_input = torch.cat([prev_embedding, context], dim=1)
203        
204        # LSTM step
205        h_state, c_state = self.lstm(lstm_input, (h_state, c_state))
206        
207        # Predict node properties
208        node_type_logits = self.node_type_output(h_state)
209        token_logits = self.token_output(h_state)
210        num_children_logits = self.num_children_output(h_state)
211        stop_logits = self.stop_output(h_state)
212        
213        return (
214            node_type_logits,
215            token_logits,
216            num_children_logits,
217            stop_logits,
218            (h_state, c_state)
219        )
220    
221    def decode_greedy(
222        self,
223        context: torch.Tensor,
224        max_nodes: int = 50
225    ) -> List[Dict]:
226        """
227        Greedy decoding to generate AST
228        
229        Returns list of node predictions
230        """
231        batch_size = context.size(0)
232        device = context.device
233        
234        # Initialize states
235        h_state = torch.zeros(batch_size, self.hidden_size, device=device)
236        c_state = torch.zeros(batch_size, self.hidden_size, device=device)
237        
238        # Start token embedding
239        prev_embedding = torch.zeros(batch_size, self.node_type_embedding.embedding_dim, device=device)
240        
241        nodes = []
242        
243        for _ in range(max_nodes):
244            # Decode step
245            (node_type_logits, token_logits, num_children_logits, 
246             stop_logits, (h_state, c_state)) = self.forward_step(
247                prev_embedding, context, h_state, c_state
248            )
249            
250            # Sample predictions
251            node_type = torch.argmax(node_type_logits, dim=1)
252            token = torch.argmax(token_logits, dim=1)
253            num_children = torch.argmax(num_children_logits, dim=1)
254            should_stop = torch.argmax(stop_logits, dim=1)
255            
256            nodes.append({
257                'node_type': node_type.item(),
258                'token': token.item(),
259                'num_children': num_children.item()
260            })
261            
262            # Update embedding for next step
263            prev_embedding = self.node_type_embedding(node_type)
264            
265            # Stop if predicted
266            if should_stop.item() == 1:
267                break
268        
269        return nodes
270
271
272class ASTacticModel(nn.Module):
273    """
274    Complete ASTactic model
275    Encodes proof context and generates tactic AST
276    """
277    
278    def __init__(
279        self,
280        vocab_size: int,
281        num_node_types: int,
282        embed_size: int = 256,
283        hidden_size: int = 512,
284        num_encoder_layers: int = 2
285    ):
286        super().__init__()
287        
288        # Encoder for proof context
289        self.context_encoder = ContextEncoder(
290            vocab_size=vocab_size,
291            embed_size=embed_size,
292            hidden_size=hidden_size,
293            num_layers=num_encoder_layers
294        )
295        
296        # Decoder for AST generation
297        self.ast_decoder = ASTDecoder(
298            num_node_types=num_node_types,
299            vocab_size=vocab_size,
300            context_size=hidden_size * 2,  # Bidirectional
301            hidden_size=hidden_size,
302            embed_size=embed_size
303        )
304        
305        self.vocab_size = vocab_size
306        self.num_node_types = num_node_types
307    
308    def forward(
309        self,
310        context_tokens: torch.Tensor,
311        target_nodes: Optional[torch.Tensor] = None
312    ) -> Dict[str, torch.Tensor]:
313        """
314        Forward pass
315        
316        Args:
317            context_tokens: [batch_size, seq_len] - tokenized proof context
318            target_nodes: Optional target AST nodes for training
319        
320        Returns:
321            Dictionary with logits and loss (if targets provided)
322        """
323        
324        # Encode context
325        context_vector = self.context_encoder(context_tokens)
326        
327        # Generate AST
328        # For training, we would use teacher forcing with target_nodes
329        # For inference, we use greedy or beam search
330        
331        if target_nodes is not None:
332            # Training mode - compute loss
333            # This would require implementing teacher forcing
334            # Simplified here
335            return {'loss': torch.tensor(0.0, device=context_tokens.device)}
336        else:
337            # Inference mode
338            predictions = self.ast_decoder.decode_greedy(context_vector)
339            return {'predictions': predictions}
340    
341    def generate_tactic(
342        self,
343        context_tokens: torch.Tensor,
344        vocabulary: TacticVocabulary,
345        max_nodes: int = 50
346    ) -> ASTNode:
347        """
348        Generate complete tactic AST from context
349        
350        Args:
351            context_tokens: Tokenized proof context
352            vocabulary: Tactic vocabulary for decoding
353            max_nodes: Maximum number of nodes in AST
354        
355        Returns:
356            Generated AST
357        """
358        self.eval()
359        
360        with torch.no_grad():
361            context_vector = self.context_encoder(context_tokens)
362            node_predictions = self.ast_decoder.decode_greedy(
363                context_vector, 
364                max_nodes=max_nodes
365            )
366        
367        # Convert predictions to AST
368        ast = self._predictions_to_ast(node_predictions, vocabulary)
369        
370        return ast
371    
372    def _predictions_to_ast(
373        self,
374        predictions: List[Dict],
375        vocabulary: TacticVocabulary
376    ) -> ASTNode:
377        """Convert list of node predictions to AST"""
378        
379        if not predictions:
380            # Default: assumption
381            return ASTNode(node_type=TacticNodeType.ASSUMPTION)
382        
383        # Build AST from predictions
384        # This is simplified - real implementation would use stack-based construction
385        
386        root_pred = predictions[0]
387        node_type = vocabulary.decode_node_type(root_pred['node_type'])
388        
389        root = ASTNode(node_type=node_type)
390        
391        # Add children based on num_children prediction
392        num_children = min(root_pred['num_children'], len(predictions) - 1)
393        
394        for i in range(1, num_children + 1):
395            if i < len(predictions):
396                child_pred = predictions[i]
397                child_type = vocabulary.decode_node_type(child_pred['node_type'])
398                
399                if child_type == TacticNodeType.IDENTIFIER:
400                    token = vocabulary.decode_token(child_pred['token'])
401                    child = ASTNode(node_type=child_type, value=token)
402                else:
403                    child = ASTNode(node_type=child_type)
404                
405                root.children.append(child)
406        
407        return root
408
409
410def create_astactic_model(
411    vocab_size: int = 10000,
412    num_node_types: int = 30,
413    hidden_size: int = 512,
414    device: str = 'cuda'
415) -> ASTacticModel:
416    """
417    Factory function to create ASTactic model
418    
419    Args:
420        vocab_size: Size of token vocabulary
421        num_node_types: Number of AST node types
422        hidden_size: Hidden dimension size
423        device: Device to load model on
424    
425    Returns:
426        Initialized ASTactic model
427    """
428    
429    model = ASTacticModel(
430        vocab_size=vocab_size,
431        num_node_types=num_node_types,
432        embed_size=256,
433        hidden_size=hidden_size,
434        num_encoder_layers=2
435    )
436    
437    model = model.to(device)
438    
439    # Count parameters
440    num_params = sum(p.numel() for p in model.parameters())
441    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
442    
443    print(f"ASTactic Model Created:")
444    print(f"  Total parameters: {num_params:,}")
445    print(f"  Trainable parameters: {trainable_params:,}")
446    print(f"  Vocabulary size: {vocab_size}")
447    print(f"  Node types: {num_node_types}")
448    print(f"  Hidden size: {hidden_size}")
449    
450    return model
451
452
453if __name__ == "__main__":
454    print("=== Testing ASTactic Model ===\n")
455    
456    # Create model
457    model = create_astactic_model(
458        vocab_size=5000,
459        num_node_types=len(TacticNodeType),
460        hidden_size=256,
461        device='cpu'
462    )
463    
464    # Test forward pass
465    batch_size = 2
466    seq_len = 20
467    
468    context_tokens = torch.randint(0, 5000, (batch_size, seq_len))
469    
470    print("\nTesting forward pass...")
471    output = model(context_tokens)
472    print(f"Output keys: {output.keys()}")
473    
474    # Test generation
475    print("\nTesting tactic generation...")
476    vocabulary = TacticVocabulary()
477    vocabulary.add_token("x")
478    vocabulary.add_token("theorem")
479    
480    generated_ast = model.generate_tactic(
481        context_tokens[0:1], 
482        vocabulary,
483        max_nodes=10
484    )
485    
486    print(f"Generated AST: {generated_ast.to_string()}")
487    print(f"AST size: {generated_ast.size()}")
488    print(f"AST depth: {generated_ast.depth()}")
489    
490    print("\n✓ ASTactic model test successful!")
491
