{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 欠損値補間の自動化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def impute(df, strategy='mean'):\n",
    "    from sklearn.preprocessing import Imputer\n",
    "    imp_num = Imputer(missing_values='NaN', strategy=strategy, axis=0)\n",
    "    # 欠損がある列を取り出す。\n",
    "    isnull_sums = df.isnull().sum() #欠損値の数を計算\n",
    "    isnull_index = []\n",
    "    for index, isnull_sum in enumerate(isnull_sums): #欠損値の数が0より多い場合のインデックスを抽出\n",
    "        if isnull_sum > 0:\n",
    "            isnull_index.append(index)\n",
    "            isnull_df = df.iloc[:, isnull_index]\n",
    "\n",
    "    display('欠損値があるデータの抽出',isnull_df)\n",
    "    print('-------------------------------')\n",
    "    display('データ型の確認',isnull_df.dtypes)\n",
    "    print('-------------------------------')\n",
    "\n",
    "    # 数字データのみの列を取り出す\n",
    "    df_nums = df.iloc[:, isnull_index].select_dtypes(include=[int, float])\n",
    "    df_not_nums = df.iloc[:, isnull_index].select_dtypes(exclude=[int, float])\n",
    "\n",
    "    if not df_nums.empty:\n",
    "        display('以下のデータを補間します。',df_nums)\n",
    "        print('-------------------------------')\n",
    "\n",
    "        # インスタンス化した imp_num に与えて、欠損値を補間\n",
    "        imputed_data = imp_num.fit_transform(df_nums)\n",
    "        print(imputed_data)\n",
    "        df_nums[:] = imputed_data\n",
    "        display('以下のデータで補間します。',df_nums)\n",
    "        print('-------------------------------')\n",
    "\n",
    "        # 元のdfに代入\n",
    "        df[df_nums.columns] = df_nums\n",
    "    display('補間完了',df)\n",
    "    print('-------------------------------')\n",
    "\n",
    "    if not df_not_nums.empty:\n",
    "        display('以下は数値データでは無いので欠損値の補間が出来ません。',df_not_nums)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### カテゴリカル・データ補間の自動化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "# カテゴリカル・データ補間の自動化\n",
    "def categorical(data, size2int, data_column):\n",
    "    data[data_column] = data[data_column].map(size2int)\n",
    "    display(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### ダミー化の自動化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ダミー化の自動化\n",
    "def dummy(data, data_column, categories):\n",
    "    # size の列のデータタイプを category に変換\n",
    "    data[data_column] = pandas.Categorical(data[data_column], categories=categories, ordered=True)\n",
    "\n",
    "    # get_dummies()関数を利用してダミー化,dummy_na=Trueの引数を与える。\n",
    "    data = pandas.get_dummies(data, columns=[data_column], dummy_na=True)\n",
    "\n",
    "    display(data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### 特徴量選択の自動化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def importance(fit_data, columns, count):\n",
    "    '''\n",
    "    fit_dataはランダムフォレストか決定木のfit\n",
    "    columnsは特徴量のラベールネーム\n",
    "    countは抽出する特徴量の数かmean（平均以上）\n",
    "    '''\n",
    "    import numpy\n",
    "    from matplotlib import pyplot\n",
    "    import japanize_matplotlib\n",
    "\n",
    "    feature_names = numpy.array(columns)\n",
    "    # 特徴重要度はfeature_importance_に格納されている\n",
    "    feature_importances = fit_data.feature_importances_\n",
    "    # ソートするが、返す値はソート前のindex、これの通りに後から並び替えたりできる\n",
    "    indices = numpy.argsort(feature_importances)\n",
    "\n",
    "    pyplot.figure(figsize=(11, 7))\n",
    "    pyplot.title('Feature Imoportances')\n",
    "\n",
    "    # barhで横向きの棒グラフの描写設定（縦はbar）,range(len(indices)):横軸9個, feature_importances[indices]:indecesの順番でプロット\n",
    "    print(range(len(indices)))\n",
    "    print(feature_importances[indices])\n",
    "    pyplot.barh(range(len(indices)), feature_importances[indices], color='b', align='center')\n",
    "    pyplot.yticks(range(len(indices)), feature_names[indices])\n",
    "    pyplot.show()\n",
    "\n",
    "    # 特徴量の重要度から決定領域をプロットするためのデータを決める\n",
    "    print(feature_names[indices])\n",
    "    if count == 'mean':\n",
    "        mu = feature_importances.mean()\n",
    "        top_culumns_name = feature_names[indices][::-1][feature_importances[indices][::-1] > mu]\n",
    "        print(top_culumns_name)\n",
    "        return top_columns_name\n",
    "    elif isinstance(count, int):\n",
    "        # [::-1]で逆順に、[:2]で最初から二番目まで取り出し\n",
    "        top_columns_name = feature_names[indices][::-1][:count]\n",
    "        print(top_columns_name)\n",
    "        return top_columns_name"
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
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
