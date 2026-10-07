"""Offline RSA and real tiny-Llama head/projection regressions."""
from pathlib import Path
import sys
import unittest
import numpy as np
import torch
from scipy.stats import spearmanr
from transformers import LlamaConfig, LlamaForCausalLM
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from concept_vectors import capture_heads, project_heads, rank_heads, rsa_features, rsa_scores, stimuli
from interventions import ActivationEngine
from activation_steer import validate_holdout


class ConceptVectorTests(unittest.TestCase):
    def test_rsa_recovers_format_invariant_head_not_format_head(self):
        examples = stimuli(['book','fork','towel'], [0,1])
        x = np.zeros((len(examples),1,3,3))
        for i,e in enumerate(examples):
            x[i,0,0,['Creative','Conventional','Effective'].index(e['Condition'])] = 1
            x[i,0,1,['numbered','bullets'].index(e['Format'])] = 1
            x[i,0,2,0] = 1  # constant/undefined RSA must not be selected
        rows=rank_heads(x,examples,permutations=19)
        self.assertEqual(rows[0]['Head'],0)
        self.assertAlmostEqual(rows[0]['Cross_RSA'],1)
        self.assertTrue(np.isnan(next(r for r in rows if r['Head']==2)['Cross_RSA']))
        self.assertEqual(rows[0]['MaxT_P'],.05)

    def test_spearman_ties_match_scipy(self):
        examples=stimuli(['book','fork'],[0,1])
        x=np.random.default_rng(2).integers(1,4,size=(len(examples),1,1,3)).astype(float)
        features,_,i,j=rsa_features(x,examples)
        labels=np.array([e['Condition'] for e in examples])
        score=rsa_scores(features,labels,i,j)[0]
        v=x[:,0,0]; v/=np.linalg.norm(v,axis=1,keepdims=True)
        expected=spearmanr(np.round((v@v.T)[i,j],12),(labels[i]==labels[j]).astype(float)).statistic
        self.assertAlmostEqual(score,expected)

    def test_audit_excludes_same_item_and_format(self):
        examples=stimuli(['shoe','umbrella'],[2])
        x=np.random.default_rng(1).normal(size=(len(examples),1,1,3))
        _,_,i,j=rsa_features(x,examples,cross_wording=False)
        self.assertTrue(all(examples[a]['Item']!=examples[b]['Item'] and examples[a]['Format']!=examples[b]['Format'] for a,b in zip(i,j)))

    def test_capture_projection_and_cleanup(self):
        model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,
              intermediate_size=32,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2)).eval()
        engine=ActivationEngine(model)
        inputs={'input_ids':torch.tensor([[1,2,3]])}
        heads=capture_heads(engine,inputs)
        self.assertEqual(tuple(heads.shape),(2,4,4))
        torch.testing.assert_close(heads[0,2],engine.capture(inputs,0,2))
        selected=[{'Layer':0,'Head':1},{'Layer':0,'Head':3}]
        projected=project_heads(engine,heads,selected)
        masked=torch.zeros(16); masked[4:8]=heads[0,1]; masked[12:]=heads[0,3]
        torch.testing.assert_close(projected,model.model.layers[0].self_attn.o_proj.weight.detach()@masked)
        self.assertFalse(model.model.layers[0].self_attn.o_proj._forward_pre_hooks)
        with self.assertRaises(Exception):
            capture_heads(engine,{'input_ids':torch.tensor([[1000]])})
        self.assertTrue(all(not l.self_attn.o_proj._forward_pre_hooks for l in model.model.layers))

    def test_audit_items_cannot_be_used_for_causal_validation(self):
        artifact={'examples':[{'Task':'Alternative Uses Task','Item':'book','Paraphrase':0}],
                  'audit_examples':[{'Task':'Alternative Uses Task','Item':'shoe','Paraphrase':2}]}
        with self.assertRaises(ValueError):
            validate_holdout(artifact,'Alternative Uses Task',['shoe'],[1],'validation')


if __name__=='__main__': unittest.main()
