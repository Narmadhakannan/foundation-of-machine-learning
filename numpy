{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "55dd0d8a-5c19-4e98-af7f-1a3ec2770c05",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 2 3 4 5]\n"
     ]
    }
   ],
   "source": [
    "\n",
    "import numpy as np\n",
    "arr = np.array([1, 2, 3, 4, 5])\n",
    "print(arr)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "67217314-1a51-41f6-809c-28b258c11a68",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Size of each element of list in bytes:  48\n",
      "Size of the whole list in bytes:  48000\n",
      "Size of each element of the Numpy array in bytes:  8\n",
      "Size of the whole Numpy array in bytes:  8000\n"
     ]
    }
   ],
   "source": [
    "import sys\n",
    "S= range(1000)\n",
    "print(\"Size of each element of list in bytes: \", sys.getsizeof(S))\n",
    "print(\"Size of the whole list in bytes: \",sys.getsizeof(S)*len(S))\n",
    "D= np.arange(1000)\n",
    "print(\"Size of each element of the Numpy array in bytes: \",D.itemsize)\n",
    "print(\"Size of the whole Numpy array in bytes: \",D.size*D.itemsize)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "16466a8d-1256-43ed-896f-3ebc35015b8c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "42\n"
     ]
    }
   ],
   "source": [
    "arr = np.array(42)\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "34b28676-6973-496d-8e37-58ad9d9c8abd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1 2 3]\n",
      " [4 5 6]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.array([[1, 2, 3], [4, 5, 6]])\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "21eb5168-1b75-42ac-a310-cb9fc4a5ebdc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[[1 2 3]\n",
      "  [4 5 6]]\n",
      "\n",
      " [[1 2 3]\n",
      "  [4 5 6]]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])\n",
    "print(arr)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "ba6bd207-9c1d-41f4-8721-21df2495e6a7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1. 1. 1. 1. 1. 1.]\n",
      "\n",
      "Data type of the array:  float64\n"
     ]
    }
   ],
   "source": [
    "ones_arr = np.ones (6)\n",
    "print(ones_arr)\n",
    "print(\"\\nData type of the array: \",ones_arr.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9c8fa9fd-9f8d-4c26-9f66-9e6aaa606821",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 1.]\n",
      " [1. 1.]\n",
      " [1. 1.]]\n",
      "\n",
      "Data type of the array:  float64\n"
     ]
    }
   ],
   "source": [
    "empty_arr = np.empty((3,2))\n",
    "print(empty_arr)\n",
    "print(\"\\nData type of the array: \",empty_arr.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "75d1716a-87f6-4f3b-a2b4-91caafab7409",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1. 0. 0.]\n",
      " [0. 1. 0.]\n",
      " [0. 0. 1.]]\n",
      "\n",
      " [[1. 0. 0. 0.]\n",
      " [0. 1. 0. 0.]\n",
      " [0. 0. 1. 0.]\n",
      " [0. 0. 0. 1.]]\n"
     ]
    }
   ],
   "source": [
    "identity_matrix = np.eye(3)\n",
    "print(identity_matrix)\n",
    "identity_matrix = np.eye(4)\n",
    "print(\"\\n\", identity_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "4d00269c-20f2-4de7-82e6-cde6ede374b0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "float64\n",
      "int32\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([1, 2, 3], dtype=np.float64)\n",
    "arr2 = np.array([1, 2, 3], dtype=np.int32)\n",
    "print(arr1.dtype)\n",
    "print(arr2.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c11a26df-db72-4886-8e8c-3b28354bdce2",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Before converting:  [1 2 3] int64\n",
      "After converting:  [1. 2. 3.] float32\n"
     ]
    }
   ],
   "source": [
    "arr = np.array([1, 2, 3])\n",
    "print(\"Before converting: \", arr, arr.dtype)\n",
    "new_arr = arr.astype('f')\n",
    "print(\"After converting: \", new_arr, new_arr.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "7eb5f9fc-923e-4bb9-9ed7-6ec09fdfeff8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "array 1\n",
      " [0 1 2 3 4 5 6 7 8 9] int64\n",
      "Array 2 \n",
      " [ 0.7341187  -0.7424923  -0.27935604  0.35118091  0.12575982  1.32810054\n",
      "  0.67180029  0.53298366 -0.98624103 -0.93671739] float64\n",
      "New Array \n",
      " [0. 1. 2. 3. 4. 5. 6. 7. 8. 9.] float64\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.arange(10)\n",
    "arr2 = np.random.randn(10)\n",
    "print(\"array 1\\n\",arr1, arr1.dtype)\n",
    "print(\"Array 2 \\n\", arr2, arr2.dtype)\n",
    "new_arr = arr1.astype(arr2.dtype)\n",
    "print(\"New Array \\n\", new_arr, new_arr.dtype)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "28e581d4-9491-49c8-9db1-2402b71ceea5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['hello' 'world' 'numpy']\n"
     ]
    }
   ],
   "source": [
    "str_arr = np.array(['hello', 'world', 'numpy'])\n",
    "print(str_arr)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a40f911c-01dc-4175-a771-8bdb2a8acfca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[ 6.  8. 10. 12.]\n",
      "[-4. -4. -4. -4.]\n",
      "[ 5. 12. 21. 32.]\n",
      "[0.2        0.33333333 0.42857143 0.5       ]\n",
      "[1.0000e+00 6.4000e+01 2.1870e+03 6.5536e+04]\n"
     ]
    }
   ],
   "source": [
    "import numpy as ud\n",
    "a = np.array([1., 2., 3., 4.])\n",
    "b = np.array([5., 6., 7., 8.])\n",
    "print(a + b) \n",
    "print(a - b)\n",
    "print(a * b)\n",
    "print(a /b) \n",
    "print(a ** b) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "a2c5c14f-dfd7-472c-8c83-9c9a8c352164",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1  2]\n",
      " [ 3  4  5]\n",
      " [ 6  7  8]\n",
      " [ 9 10 11]]\n",
      "\n",
      " [[ 0  4  8]\n",
      " [ 1  5  9]\n",
      " [ 2  6 10]\n",
      " [ 3  7 11]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as up\n",
    "arr = np.arange(12)\n",
    "newarr = arr.reshape(4, 3)\n",
    "print(newarr)\n",
    "newarr = arr.reshape((4, 3),order='F')\n",
    "print(\"\\n\", newarr)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "96c669d1-be68-4a1d-b9da-c3564b3688a8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 1  2  3  4  9 10]\n",
      " [ 5  6  7  8 11 12]]\n",
      "\n",
      " [[[ 1  2]\n",
      "  [ 3  4]]\n",
      "\n",
      " [[ 9 10]\n",
      "  [ 5  6]]\n",
      "\n",
      " [[ 7  8]\n",
      "  [11 12]]]\n",
      "\n",
      " reshape(3,-1) \n",
      " [[ 1  2  3  4]\n",
      " [ 9 10  5  6]\n",
      " [ 7  8 11 12]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as ud\n",
    "arr = np.array([[1, 2, 3, 4,9,10], [5, 6, 7, 8,11,12]])\n",
    "print(arr)\n",
    "newarr = arr.reshape(3, 2, 2)\n",
    "print(\"\\n\", newarr)\n",
    "newarr1 = newarr.reshape(3, -1)\n",
    "print(\"\\n reshape(3,-1) \\n\", newarr1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "22bad327-2087-4e4b-85f3-80b593d3811f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original array:\n",
      " [[10 20]\n",
      " [30 40]]\n",
      "Flattened array:\n",
      " [999  20  30  40]\n",
      "Original array:\n",
      " [[10 20]\n",
      " [30 40]]\n",
      "Raveled array:\n",
      " [ 10 888  30  40]\n",
      "Original array:\n",
      " [[ 10 888]\n",
      " [ 30  40]]\n"
     ]
    }
   ],
   "source": [
    "original_array = np.array([[10, 20], [30, 40]])\n",
    "print(\"Original array:\\n\", original_array)\n",
    "flattened_array = original_array.flatten()\n",
    "flattened_array [0] = 999\n",
    "print(\"Flattened array:\\n\", flattened_array)\n",
    "print(\"Original array:\\n\", original_array)\n",
    "raveled_array = original_array.ravel()\n",
    "raveled_array[1] = 888\n",
    "print(\"Raveled array:\\n\", raveled_array)\n",
    "print(\"Original array:\\n\", original_array)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "ef4da367-e245-404a-a4b4-b7ab254ae5fa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Greater than:\n",
      " [[False  True]\n",
      " [False False]]\n",
      "\n",
      "Less than:\n",
      " [[ True False]\n",
      " [False  True]]\n",
      "\n",
      "Equal to:\n",
      " [[False False]\n",
      " [ True False]]\n",
      "\n",
      "Array-wise comparison 1: True\n",
      "Array-wise comparison 2: False\n"
     ]
    }
   ],
   "source": [
    "arr1 = np.array([[1, 2], [3, 4]])\n",
    "arr2 = np.array([[2, 1], [3, 5]])\n",
    "greater_than = arr1 > arr2\n",
    "less_than = arr1 < arr2\n",
    "equal_to = arr1 == arr2\n",
    "print(\"Greater than:\\n\", greater_than)\n",
    "print(\"\\nLess than:\\n\", less_than)\n",
    "print(\"\\nEqual to:\\n\", equal_to)\n",
    "arr3 = np.array([1, 2, 3])\n",
    "arr4 = np.array([1, 2, 3])\n",
    "arr5 = np.array([1, 2, 4])\n",
    "result1 = np.array_equal(arr3, arr4) # True\n",
    "result2 = np.array_equal(arr3, arr5) # False\n",
    "print(\"\\nArray-wise comparison 1:\", result1)\n",
    "print(\"Array-wise comparison 2:\", result2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "9cf5ca8c-4590-4b81-b4cd-8abf33ff373d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x=  [0 1 2 3 4 5 6 7 8 9]\n",
      "\n",
      " x[:5]=  [0 1 2 3 4]\n",
      "\n",
      " x[5:]=  [5 6 7 8 9]\n",
      "\n",
      " x[4:7]=  [4 5 6]\n",
      "\n",
      " x[::2]=  [0 2 4 6 8]\n",
      "\n",
      " x[1::2]=  [1 3 5 7 9]\n",
      "\n",
      " x[::-1]=  [9 8 7 6 5 4 3 2 1 0]\n",
      "\n",
      " x[5::-2]=  [5 3 1]\n"
     ]
    }
   ],
   "source": [
    "x = np.arange (10)\n",
    "print(\"x= \",x)\n",
    "print(\"\\n x[:5]= \",x[:5]) \n",
    "print(\"\\n x[5:]= \",x[5:])\n",
    "print(\"\\n x[4:7]= \",x[4:7]) \n",
    "print(\"\\n x[::2]= \",x[::2]) \n",
    "print(\"\\n x[1::2]= \",x[1::2]) \n",
    "print(\"\\n x[::-1]= \",x[::-1]) \n",
    "print(\"\\n x[5::-2]= \",x[5::-2]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "0acde087-a726-427d-bd2f-798cc197272e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 3 4 5 6 7 8 9]\n",
      "\n",
      " After broadcasting1 [0 1 2 3 4 2 2 2 2 2]\n",
      "\n",
      " After broadcasting2 [100   1 100   3 100   2 100   2 100   2]\n"
     ]
    }
   ],
   "source": [
    "x = np.arange (10)\n",
    "print(x)\n",
    "x[5:] = 2\n",
    "print(\"\\n After broadcasting1\",x)\n",
    "x[::2] = 100\n",
    "print(\"\\n After broadcasting2\",x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "9e68f369-c687-48c8-abf2-87e872052322",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[12  5  2  4]\n",
      "1\n",
      "8\n",
      "7\n",
      "[12  5  2  4]\n"
     ]
    }
   ],
   "source": [
    "x2 = np.array([[12, 5, 2, 4],[7, 6, 8, 8],[1, 6, 7, 7]])\n",
    "print(x2[0]) \n",
    "print(x2[2,0]) \n",
    "print(x2[1] [2]) \n",
    "print(x2[2,-1]) \n",
    "print(x2[0]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "1a86479f-7843-4b25-b0b1-86bdfe2e4f7c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "x2=  [[12  5  2  4]\n",
      " [ 7  6  8  8]\n",
      " [ 1  6  7  7]]\n",
      "\n",
      " x2[:2, :3] = \n",
      " [[12  5  2]\n",
      " [ 7  6  8]]\n",
      "\n",
      " x2[:3, ::2]=\n",
      "  [[12  2]\n",
      " [ 7  8]\n",
      " [ 1  7]]\n",
      "\n",
      " x2[::-1,::-1]= \n",
      " [[ 7  7  6  1]\n",
      " [ 8  8  6  7]\n",
      " [ 4  2  5 12]]\n",
      "\n",
      " x2[::-1]= \n",
      " [[ 1  6  7  7]\n",
      " [ 7  6  8  8]\n",
      " [12  5  2  4]]\n"
     ]
    }
   ],
   "source": [
    "x2 = np.array([[12, 5, 2, 4], [7, 6, 8, 8], [1, 6, 7, 7]])\n",
    "print(\"x2= \", x2)\n",
    "print(\"\\n x2[:2, :3] = \\n\",x2[:2, :3]) \n",
    "print(\"\\n x2[:3, ::2]=\\n \",x2[:3, ::2])\n",
    "print(\"\\n x2[::-1,::-1]= \\n\",x2[::-1, ::-1]) \n",
    "print(\"\\n x2[::-1]= \\n\",x2[::-1]) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "73828bf6-0b18-4ce1-a039-f1d3cc7253a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a= \n",
      " [[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]]\n",
      "b= \n",
      " [[False False False False]\n",
      " [False  True  True  True]\n",
      " [ True  True  True  True]]\n",
      "a[b]= \n",
      " [ 5  6  7  8  9 10 11]\n",
      "a= \n",
      " [[0 1 2 3]\n",
      " [4 1 1 1]\n",
      " [1 1 1 1]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as up\n",
    "a = np.arange(12).reshape(3,4)\n",
    "print(\"a= \\n\", a)\n",
    "b = a > 4\n",
    "print(\"b= \\n\", b)\n",
    "print(\"a[b]= \\n\",a[b]) \n",
    "a[b] = 1\n",
    "print(\"a= \\n\",a)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "1e4ddcea-c76c-4aac-aaf0-a9f702b5033d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['Bob' 'Joe' 'Will' 'Bob' 'Will' 'Joe' 'Joe']\n",
      "data= \n",
      " [[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]\n",
      " [12 13 14 15]\n",
      " [16 17 18 19]\n",
      " [20 21 22 23]\n",
      " [24 25 26 27]]\n",
      "names bool= \n",
      " [ True False False  True False False False]\n",
      "names == 'Bob' \n",
      " [[ 0  1  2  3]\n",
      " [12 13 14 15]]\n",
      "names == 'Bob',2: \n",
      " [[ 2  3]\n",
      " [14 15]]\n",
      "names == 'Bob',3 \n",
      " [ 3 15]\n"
     ]
    }
   ],
   "source": [
    "names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])\n",
    "data = np.arange(28).reshape(7, 4)\n",
    "print(names)\n",
    "print(\"data= \\n\",data)\n",
    "print(\"names bool= \\n\",names == 'Bob') \n",
    "print(\"names == 'Bob' \\n\", data [names == 'Bob']) \n",
    "print(\"names == 'Bob',2: \\n\", data[names == 'Bob', 2:]) \n",
    "print(\"names == 'Bob',3 \\n\", data[names == 'Bob', 3]) \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "b0e91a11-1169-46ac-acf0-78cdcec5435a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "arr= \n",
      " [[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]\n",
      " [12 13 14 15]\n",
      " [16 17 18 19]\n",
      " [20 21 22 23]\n",
      " [24 25 26 27]\n",
      " [28 29 30 31]]\n",
      "\n",
      "arr[[1, 5, 7, 2]]= \n",
      " [[ 4  5  6  7]\n",
      " [20 21 22 23]\n",
      " [28 29 30 31]\n",
      " [ 8  9 10 11]]\n",
      "\n",
      "arr[[1, 5, 7, 2], [0, 3, 1, 2]]= \n",
      " [ 4 23 29 10]\n",
      "\n",
      "arr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]= \n",
      " [[ 4  7  5  6]\n",
      " [20 23 21 22]\n",
      " [28 31 29 30]\n",
      " [ 8 11  9 10]]\n",
      "\n",
      " arr[[-3, -5, -7]]\n",
      " [[20 21 22 23]\n",
      " [12 13 14 15]\n",
      " [ 4  5  6  7]]\n",
      "\n",
      "arr= \n",
      " [[ 0  1  2  3]\n",
      " [ 1  1  1  1]\n",
      " [ 1  1  1  1]\n",
      " [12 13 14 15]\n",
      " [16 17 18 19]\n",
      " [ 1  1  1  1]\n",
      " [24 25 26 27]\n",
      " [ 1  1  1  1]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(32).reshape(8, 4)\n",
    "print(\"arr= \\n\",arr)\n",
    "print(\"\\narr[[1, 5, 7, 2]]= \\n\",arr[[1, 5, 7, 2]])\n",
    "print(\"\\narr[[1, 5, 7, 2], [0, 3, 1, 2]]= \\n\",arr[[1, 5, 7, 2], [0, 3, 1, 2]])\n",
    "print(\"\\narr[[1, 5, 7, 2]][:, [0, 3, 1, 2]]= \\n\",arr[[1, 5, 7, 2]][:, [0, 3, 1,2]])\n",
    "print(\"\\n arr[[-3, -5, -7]]\\n\",arr[[-3, -5, -7]])\n",
    "arr[[1, 5, 7, 2]] = 1\n",
    "print(\"\\narr= \\n\",arr)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "2dd409b1-48cd-41d8-8204-7ffc262956dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]]\n",
      "Transpose using .T\n",
      "[[ 0  4  8]\n",
      " [ 1  5  9]\n",
      " [ 2  6 10]\n",
      " [ 3  7 11]]\n",
      "Transpose using the transpose() function\n"
     ]
    }
   ],
   "source": [
    "import numpy as up\n",
    "arr = np.arange(12).reshape((3, 4))\n",
    "print(arr)\n",
    "print(\"Transpose using .T\")\n",
    "print(arr.T)\n",
    "print(\"Transpose using the transpose() function\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "19a5c2eb-c4ad-402c-8ce6-a26175cd7dc6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  4  8]\n",
      " [ 1  5  9]\n",
      " [ 2  6 10]\n",
      " [ 3  7 11]]\n",
      "[[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]]\n"
     ]
    }
   ],
   "source": [
    "arr1=np.transpose(arr,(1,0))\n",
    "print(arr1)\n",
    "arr2=np.transpose(arr,(0,1))\n",
    "print(arr2)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "9ed3360f-07b3-4694-ac40-01144ad2ac9f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dot product of arr and arr2:\n",
      "[[ 14  38  62]\n",
      " [ 38 126 214]\n",
      " [ 62 214 366]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as ud\n",
    "arr = np.arange(12).reshape((3, 4))\n",
    "print(\"Dot product of arr and arr2:\")\n",
    "print(np.dot(arr, arr.T))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "fa81cac2-ae5d-403a-9661-c4547f77431d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0  1  2  3]\n",
      " [ 4  5  6  7]\n",
      " [ 8  9 10 11]]\n",
      "Swap axes 0 and 1:\n",
      "[[ 0  4  8]\n",
      " [ 1  5  9]\n",
      " [ 2  6 10]\n",
      " [ 3  7 11]]\n"
     ]
    }
   ],
   "source": [
    "arr = np.arange(12).reshape((3, 4))\n",
    "print(arr)\n",
    "print(\"Swap axes 0 and 1:\")\n",
    "print(np.swapaxes (arr, 0, 1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "3c023250-5f3d-4a2d-b4aa-3bd6fef29a42",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Result of element-wise multiplication:\n",
      "[[ 5 12]\n",
      " [21 32]]\n",
      "Result of matrix multiplication:\n",
      "[[19 22]\n",
      " [43 50]]\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "arr1 = np.array([[1, 2],\n",
    "[3, 4]])\n",
    "arr2 = np.array([[5, 6],\n",
    "[7, 8]])\n",
    "result = np.multiply(arr1, arr2)\n",
    "print(\"Result of element-wise multiplication:\")\n",
    "print(result)\n",
    "result = np.matmul(arr1, arr2)\n",
    "print(\"Result of matrix multiplication:\")\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "bf5ead6e-6e03-4e8e-a062-7da06b3f0033",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Result of matrix multiplication:\n",
      "[[19 22]\n",
      " [43 50]]\n"
     ]
    }
   ],
   "source": [
    "result = np.dot(arr1, arr2)\n",
    "print(\"Result of matrix multiplication:\")\n",
    "print(result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0570b1f1-7ebf-4da0-bd98-9c054cf8b2b3",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
