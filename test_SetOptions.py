import pytest
from SetOptions import set_options  # Assuming set_options is in a file named SetOptions.py

def test_set_options():
    optns = {'methodXi': None}
    
    # Test case 1: Expect methodXi to be 'CE'
    result = set_options([[1, 3, 5], [2, 4]], [[1, 3, 5], [2, 4]], optns)
    assert result['methodXi'] == 'CE'
    
    # Test case 2: Expect methodXi to be 'IN'
    result = set_options([list(range(1, 11)), list(range(1, 11))], [list(range(1, 11)), list(range(1, 11))], optns)
    assert result['methodXi'] == 'IN'
    
    # Test case 3: Expect kernel to be 'epan'
    result = set_options([list(range(1, 11)), list(range(1, 11))], [list(range(1, 11)), list(range(1, 11))], optns)
    assert result['kernel'] == 'epan'
    
    # Test case 4: Expect methodXi to be 'CE'
    result = set_options([[1, 2, 3, 4, 5], [1, 2, 3, 4]], [[1, 2, 3, 4, 5], [1, 2, 3, 4]], optns)
    assert result['methodXi'] == 'CE'

if __name__ == '__main__':
    pytest.main()
