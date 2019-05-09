{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# WEEK6 실전 분석\n",
    "## 숫자 손글씨 분류하기\n",
    "데이터 원작자 소개 MNIST http://yann.lecun.com/exdb/mnist/\n",
    "\n",
    "---\n",
    "\n",
    "## 문제 정의\n",
    "28x28 픽셀의 손글씨 숫자 이미지를 입력받아 실제로 의미하는 숫자를 인식한다.\n",
    "\n",
    "## 가설 수립\n",
    "28x28 픽셀 데이터, 즉 784종의 특징 데이터를 구성한 후 머신러닝을 통해 실제로 어떤 숫자인지 추측할 수 있다.\n",
    "\n",
    "## 목표\n",
    "28x28 사이즈의 숫자 손글씨 이미지로 부터 label값을 얻어낸다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "## 데이터 구성\n",
    "\n",
    "**784개 입력 데이터** 28x28 픽셀이므로 총 784칸의 픽셀 값 정보  \n",
    "**출력 데이터** label(어떤 숫자인가?)  \n",
    "**개수** 10,000개"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 데이터 다운로드\n",
    "[데이터 다운로드](http://bit.ly/코알라_DS_6주차_데이터)\n",
    "-교육적 의도로 기존 DATA에서 일부 변형되었습니다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 분석 가이드\n",
    "### [단계1. 데이터 불러오기](#1.-데이터-불러오기)\n",
    "### [단계2. EDA & Feature Engineering](#2.-EDA-&-Feature-Engineering)\n",
    "### [단계3. Dataset 구성하기](#3.-Dataset-구성하기)\n",
    "### [단계4. 모델링과 학습](#4.-모델링과-학습)\n",
    "### [단계5. 모델 검증](#5.-모델-검증)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 1. 데이터 불러오기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:50.648446Z",
     "start_time": "2019-03-16T02:23:50.177642Z"
    },
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:52.160867Z",
     "start_time": "2019-03-16T02:23:50.651603Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pixel 1,1</th>\n",
       "      <th>pixel 1,2</th>\n",
       "      <th>pixel 1,3</th>\n",
       "      <th>pixel 1,4</th>\n",
       "      <th>pixel 1,5</th>\n",
       "      <th>pixel 1,6</th>\n",
       "      <th>pixel 1,7</th>\n",
       "      <th>pixel 1,8</th>\n",
       "      <th>pixel 1,9</th>\n",
       "      <th>pixel 1,10</th>\n",
       "      <th>...</th>\n",
       "      <th>pixel 28,20</th>\n",
       "      <th>pixel 28,21</th>\n",
       "      <th>pixel 28,22</th>\n",
       "      <th>pixel 28,23</th>\n",
       "      <th>pixel 28,24</th>\n",
       "      <th>pixel 28,25</th>\n",
       "      <th>pixel 28,26</th>\n",
       "      <th>pixel 28,27</th>\n",
       "      <th>pixel 28,28</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 785 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   pixel 1,1  pixel 1,2  pixel 1,3  pixel 1,4  pixel 1,5  pixel 1,6  \\\n",
       "0        0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "1        0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "2        0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "3        0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "4        0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "\n",
       "   pixel 1,7  pixel 1,8  pixel 1,9  pixel 1,10  ...    pixel 28,20  \\\n",
       "0        0.0        0.0        0.0         0.0  ...            0.0   \n",
       "1        0.0        0.0        0.0         0.0  ...            0.0   \n",
       "2        0.0        0.0        0.0         0.0  ...            0.0   \n",
       "3        0.0        0.0        0.0         0.0  ...            0.0   \n",
       "4        0.0        0.0        0.0         0.0  ...            0.0   \n",
       "\n",
       "   pixel 28,21  pixel 28,22  pixel 28,23  pixel 28,24  pixel 28,25  \\\n",
       "0          0.0          0.0          0.0          0.0          0.0   \n",
       "1          0.0          0.0          0.0          0.0          0.0   \n",
       "2          0.0          0.0          0.0          0.0          0.0   \n",
       "3          0.0          0.0          0.0          0.0          0.0   \n",
       "4          0.0          0.0          0.0          0.0          0.0   \n",
       "\n",
       "   pixel 28,26  pixel 28,27  pixel 28,28  label  \n",
       "0          0.0          0.0          0.0      4  \n",
       "1          0.0          0.0          0.0      8  \n",
       "2          0.0          0.0          0.0      8  \n",
       "3          0.0          0.0          0.0      7  \n",
       "4          0.0          0.0          0.0      4  \n",
       "\n",
       "[5 rows x 785 columns]"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('data/digit.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 2. EDA & Feature Engineering"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:54.215662Z",
     "start_time": "2019-03-16T02:23:52.163001Z"
    },
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline\n",
    "\n",
    "import seaborn as sns\n",
    "sns.set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.010246Z",
     "start_time": "2019-03-16T02:23:54.219364Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>pixel 1,1</th>\n",
       "      <th>pixel 1,2</th>\n",
       "      <th>pixel 1,3</th>\n",
       "      <th>pixel 1,4</th>\n",
       "      <th>pixel 1,5</th>\n",
       "      <th>pixel 1,6</th>\n",
       "      <th>pixel 1,7</th>\n",
       "      <th>pixel 1,8</th>\n",
       "      <th>pixel 1,9</th>\n",
       "      <th>pixel 1,10</th>\n",
       "      <th>...</th>\n",
       "      <th>pixel 28,20</th>\n",
       "      <th>pixel 28,21</th>\n",
       "      <th>pixel 28,22</th>\n",
       "      <th>pixel 28,23</th>\n",
       "      <th>pixel 28,24</th>\n",
       "      <th>pixel 28,25</th>\n",
       "      <th>pixel 28,26</th>\n",
       "      <th>pixel 28,27</th>\n",
       "      <th>pixel 28,28</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>...</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.000000</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>10000.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000480</td>\n",
       "      <td>0.000239</td>\n",
       "      <td>0.000050</td>\n",
       "      <td>0.000025</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.453400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.017804</td>\n",
       "      <td>0.013588</td>\n",
       "      <td>0.003535</td>\n",
       "      <td>0.002500</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.884451</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>4.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>7.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>...</td>\n",
       "      <td>0.988281</td>\n",
       "      <td>0.988281</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>9.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>8 rows × 785 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       pixel 1,1  pixel 1,2  pixel 1,3  pixel 1,4  pixel 1,5  pixel 1,6  \\\n",
       "count    10000.0    10000.0    10000.0    10000.0    10000.0    10000.0   \n",
       "mean         0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "std          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "min          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "25%          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "50%          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "75%          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "max          0.0        0.0        0.0        0.0        0.0        0.0   \n",
       "\n",
       "       pixel 1,7  pixel 1,8  pixel 1,9  pixel 1,10      ...       \\\n",
       "count    10000.0    10000.0    10000.0     10000.0      ...        \n",
       "mean         0.0        0.0        0.0         0.0      ...        \n",
       "std          0.0        0.0        0.0         0.0      ...        \n",
       "min          0.0        0.0        0.0         0.0      ...        \n",
       "25%          0.0        0.0        0.0         0.0      ...        \n",
       "50%          0.0        0.0        0.0         0.0      ...        \n",
       "75%          0.0        0.0        0.0         0.0      ...        \n",
       "max          0.0        0.0        0.0         0.0      ...        \n",
       "\n",
       "        pixel 28,20   pixel 28,21   pixel 28,22   pixel 28,23  pixel 28,24  \\\n",
       "count  10000.000000  10000.000000  10000.000000  10000.000000      10000.0   \n",
       "mean       0.000480      0.000239      0.000050      0.000025          0.0   \n",
       "std        0.017804      0.013588      0.003535      0.002500          0.0   \n",
       "min        0.000000      0.000000      0.000000      0.000000          0.0   \n",
       "25%        0.000000      0.000000      0.000000      0.000000          0.0   \n",
       "50%        0.000000      0.000000      0.000000      0.000000          0.0   \n",
       "75%        0.000000      0.000000      0.000000      0.000000          0.0   \n",
       "max        0.988281      0.988281      0.250000      0.250000          0.0   \n",
       "\n",
       "       pixel 28,25  pixel 28,26  pixel 28,27  pixel 28,28         label  \n",
       "count      10000.0      10000.0      10000.0      10000.0  10000.000000  \n",
       "mean           0.0          0.0          0.0          0.0      4.453400  \n",
       "std            0.0          0.0          0.0          0.0      2.884451  \n",
       "min            0.0          0.0          0.0          0.0      0.000000  \n",
       "25%            0.0          0.0          0.0          0.0      2.000000  \n",
       "50%            0.0          0.0          0.0          0.0      4.000000  \n",
       "75%            0.0          0.0          0.0          0.0      7.000000  \n",
       "max            0.0          0.0          0.0          0.0      9.000000  \n",
       "\n",
       "[8 rows x 785 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.434078Z",
     "start_time": "2019-03-16T02:23:57.012880Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x1a2228cac8>"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 360x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.catplot(data=df, x='label', kind='count')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 이미지 살펴보기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.618588Z",
     "start_time": "2019-03-16T02:23:57.435991Z"
    },
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "numbers = df.drop(['label'], axis=1)\n",
    "\n",
    "nth = 0 # 0 ~ 9999까지 바꾸면서 살펴보세요\n",
    "img = np.reshape(numbers.iloc[nth].values, [28, 28])\n",
    "plt.imshow(img)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 3. Dataset 구성하기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.640714Z",
     "start_time": "2019-03-16T02:23:57.620888Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(10000, 784) (10000,)\n"
     ]
    }
   ],
   "source": [
    "train_data = df.drop('label', axis=1)\n",
    "target_data = df['label']\n",
    "\n",
    "print(train_data.shape, target_data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.822160Z",
     "start_time": "2019-03-16T02:23:57.643733Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(10000, 784) (8000, 784) (2000, 784)\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "x_train, x_test, y_train, y_test = train_test_split(train_data, target_data, test_size=0.2) \n",
    "\n",
    "print(train_data.shape, x_train.shape, x_test.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# 4. 모델링과 학습"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:23:57.946490Z",
     "start_time": "2019-03-16T02:23:57.824491Z"
    },
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "forest = RandomForestClassifier(n_estimators=100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:24:01.799069Z",
     "start_time": "2019-03-16T02:23:57.949721Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',\n",
       "            max_depth=None, max_features='auto', max_leaf_nodes=None,\n",
       "            min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "            min_samples_leaf=1, min_samples_split=2,\n",
       "            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,\n",
       "            oob_score=False, random_state=None, verbose=0,\n",
       "            warm_start=False)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# train 데이터 학습\n",
    "forest.fit(x_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:24:02.154900Z",
     "start_time": "2019-03-16T02:24:01.801111Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training set accuracy: 1.0\n"
     ]
    }
   ],
   "source": [
    "print('training set accuracy:', forest.score(x_train, y_train))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. 모델 검증"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:24:02.237910Z",
     "start_time": "2019-03-16T02:24:02.157637Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "test set accuracy: 0.947\n"
     ]
    }
   ],
   "source": [
    "print('test set accuracy:', forest.score(x_test, y_test)) # 인식정확도 출력"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 실제 예측 결과물 살펴보기"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:24:02.843258Z",
     "start_time": "2019-03-16T02:24:02.240200Z"
    },
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "인식된 숫자는 0 입니다.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "인식된 숫자는 7 입니다.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "인식된 숫자는 8 입니다.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "인식된 숫자는 3 입니다.\n"
     ]
    }
   ],
   "source": [
    "# random으로 픽하기\n",
    "import random\n",
    "for i in range(4):\n",
    "    n = random.randrange(0, len(x_test))\n",
    "\n",
    "    img = np.reshape(x_test.iloc[n].values, [28, 28])\n",
    "    plt.imshow(img)\n",
    "    plt.show()\n",
    "    \n",
    "    result = forest.predict([x_test.iloc[n].values])[0]\n",
    "    print(\"인식된 숫자는\", result, \"입니다.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---\n",
    "# Challenge - Support Vector Machine 사용해보기\n",
    "## SVR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "ExecuteTime": {
     "end_time": "2019-03-16T02:25:15.445605Z",
     "start_time": "2019-03-16T02:24:02.846080Z"
    },
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Volumes/External1/Envs/Conda/ds/lib/python3.6/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.\n",
      "  \"avoid this warning.\", FutureWarning)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training set accuracy: 0.924625\n",
      "test set accuracy: 0.9145\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "model = SVC()\n",
    "model.fit(x_train, y_train)\n",
    "\n",
    "print('training set accuracy:', model.score(x_train, y_train))\n",
    "print('test set accuracy:', model.score(x_test, y_test))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
