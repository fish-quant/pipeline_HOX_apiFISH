import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthesis.synthesize import Synthesis
import numpy as np



def test_colocalisation_analysis():
    """put together points from two different rna images wich are very close.
    """
    x  = np.array([[1, 1],[5, 3]])
    y  = np.array([[1, 1],[1, 10], [10, 10]])
    st = Synthesis()
    list_gene1_only, list_gene2_only, list_gene1_gene2 = st.colocalization_analysis(x,y)
    
    assert np.array_equal(list_gene1_only[0], np.array([5, 3])) 
    assert np.array_equal(list_gene2_only[0], np.array([1, 10]))
    assert np.array_equal(list_gene2_only[1], np.array([10, 10]))
    assert np.array_equal(list_gene1_gene2[0][0], np.array([1, 1]))
    assert np.array_equal(list_gene1_gene2[0][1], np.array([1, 1]))



