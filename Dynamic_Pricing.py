import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df_ = pd.read_csv ("/Users/ozanguner/PycharmProjects/son_dsmlbc/datasets/pricing.csv", sep=";")
df = df_.copy ()
pd.set_option ("display.max_columns", None)
pd.set_option ('display.expand_frame_repr', False)


def check_data(dataframe, head=5):
    print ("####### SHAPE #######")
    print (dataframe.shape)
    print ("####### INFO #######")
    print (dataframe.info ())
    print ("####### DESCRIBE #######")
    print (dataframe.describe ([0.01, 0.1, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]))
    print ("####### NA VALUES #######")
    print (dataframe.isnull ().sum ())
    print ("####### FIRST {} ROWS #######".format (head))
    print (dataframe.head (head))


def defining_quantile_th(dataframe, col, quantilelow=0, quantileup=1):
    up_ratio = (len (dataframe[dataframe[col] > dataframe[col].quantile (quantileup)]) / dataframe[col].shape[0]) * 100
    low_ratio = (len (dataframe[dataframe[col] < dataframe[col].quantile (quantilelow)]) / dataframe[col].shape[
        0]) * 100
    total_ratio = up_ratio + low_ratio
    print (
        "{} - {} aralığında quantile belirlendiğinde, müşterilerin % {} kadarının verisi törpülenmiş olacak.".format (
            quantilelow, quantileup, round (total_ratio, 2)))


def outliers_threshold(dataframe, col, low_ratio, up_ratio):
    threshold_range = dataframe[col].quantile (up_ratio) - dataframe[col].quantile (low_ratio)
    up_limit = dataframe[col].quantile (up_ratio) + 1.5 * threshold_range
    low_limit = dataframe[col].quantile (low_ratio) - 1.5 * threshold_range
    dataframe.loc[dataframe[col] > up_limit, "price"] = up_limit
    dataframe.loc[dataframe[col] < low_limit, "price"] = low_limit


check_data (df)


### AYKIRI GÖZLEM KONTROLLERİ
defining_quantile_th (df, "price", 0.05,
                      0.95)  # Tüm gözlemlerin % 5'i, price değişkeninin %95. değerinden yukarıda. Bu nedenle daha sağlıklı bir sonuç elde etmek için aykırı gözlemleri törpülemeliyiz.

df["price"].value_counts ().sort_values (ascending=False)

# Aykırı gözlemlere grafiksel bakış. Dağılımlarını gözlemliyoruz.
sns.distplot (df["price"])
plt.show ();

sns.boxplot (x="category_id", y="price", data=df)
plt.show ();

### Aykırı gözlemlerin törpülenmesi
outliers_threshold (df, "price", 0.05, 0.95)

check_data (df)

df["price"].value_counts ().sort_values (ascending=False)

sns.distplot (df["price"])
plt.show ();

df.groupby ("category_id").agg ({"price": ["count", "median", "mean", "std", "min", "max"]})  # Kategori ortalamaları arasındaki ilişkilere bakalım

sns.boxplot (x="category_id", y="price", data=df)
plt.ylim ((20, 80))
plt.show ();


# Normallik Varsayımı Kontrolü

def normality_test(dataframe, col, class_col):
    # H0 : normal distribution
    # H1 : nonnormal distribution
    from scipy.stats import shapiro
    category_ids = list (set (dataframe[class_col]))
    df = pd.DataFrame()
    nonnormal = []
    normal = []
    for i in category_ids:
        ttest, p_value = shapiro (dataframe.loc[dataframe[class_col] == i, col])
        if p_value < 0.05:
            df = df.append({"Category_ID":str(i),"Distribution": "Nonnormal Distribution", "Hypothesis_Situation":"H0 was Rejected"}, ignore_index=True)
            nonnormal.append(i)
        else:
            df = df.append ({"Category_ID": str (i), "Distribution": "Normal Distribution", "Hypothesis_Situation":"H0 was Failed to Reject"}, ignore_index=True)
            normal.append(i)
    return nonnormal, normal, df

nonnormal_list, normal_list, normality_df = normality_test (df, "price","category_id")  # Bütün kategorilerin price ortalamaları normal dağılmamış, bu nedenle direkt olarak Nonparametrik test uygulayabiliriz. Varyans homojenliğini kontrol etmemize gerek kalmadı.

normality_df

# İki grubun fiyat ortalamalarının anlamlılığına bakalım;
# Nonparametrik Bağımsız İki Örneklem T Testi

def nonparametric_ind_ab_test(dataframe, col, class_col, nonnormally_list):
    # H0: There is no difference between the average  of two groups
    # H1: There is difference between the average of two groups
    from scipy.stats import mannwhitneyu
    different_avg = []
    similar_avg = []
    for i in range (len (nonnormally_list)):
        for k in range (i + 1, len (nonnormally_list)):
            ttest_lev, p_value_lev = mannwhitneyu ((dataframe.loc[dataframe[class_col] == nonnormally_list[i], col]),
                                                   (dataframe.loc[dataframe[class_col] == nonnormally_list[k], col]))
            if p_value_lev < 0.05:
                different_avg.append ((nonnormally_list[i], nonnormally_list[k]))
            else:
                similar_avg.append ((nonnormally_list[i], nonnormally_list[k]))
    return similar_avg, different_avg


sim_avg, diff_avg = nonparametric_ind_ab_test (df, "price", "category_id", nonnormal_list)

df.groupby ("category_id").agg ({"price": ["count", "median", "mean", "std", "min", "max"]})

#  1. Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki olarak ifade ediniz.

diff_avg  # İstatistiksel olarak, aralarında anlamlı farklılıklar bulunan kategori id'ler.


def comparing(dataframe, col, class_col, list_):
    for i in list_:
        new_df = pd.concat ([dataframe[dataframe[class_col] == i[0]], dataframe[dataframe[class_col] == i[1]]],
                            axis=0).reset_index (drop=True)
        print (new_df.groupby (class_col).agg ({col: ["mean", "median", "std"]}))


comparing (df, "price", "category_id", diff_avg)  # Aralarında anlamlı fark bulunanların istatistiki karşılaştırmaları

#  2. İlk soruya bağlı olarak item'ın fiyatı ne olmalıdır? Nedenini açıklayınız?


# 4 Farklı Yaklaşıma Göre Güven Aralığı Hesaplaması
from statsmodels.stats.api import DescrStatsW

similar_categories = set ()
for i in sim_avg:
    similar_categories.add (i[0])
    similar_categories.add (i[1])

different_categories = set (df["category_id"].unique()) - similar_categories

different_categories


def price_df(dataframe, low_price, high_price, median_price, method_name):
    mean_price = (low_price + high_price) / 2
    income_p_mean = dataframe.loc[dataframe["price"] >= mean_price].shape[0] * mean_price
    income_p_median = dataframe.loc[dataframe["price"] >= median_price].shape[0] * median_price
    income_p_lower = dataframe.loc[dataframe["price"] >= low_price].shape[0] * low_price
    income_p_upper = dataframe.loc[dataframe["price"] >= high_price].shape[0] * high_price
    df = pd.DataFrame ({"Income_according_to_mean": income_p_mean,
                        "Mean_price": [mean_price],
                        "Mean_count": [dataframe.loc[dataframe["price"] >= mean_price].shape[0]],
                        "Income_according_to_median": income_p_median,
                        "Median_price":[median_price],
                        "Median_count":[dataframe.loc[dataframe["price"] >= median_price].shape[0]],
                        "Income_according_to_low_price": income_p_lower,
                        "Lower_price": [low_price],
                        "Low_price_count": [dataframe.loc[dataframe["price"] >= low_price].shape[0]],
                        "Income_according_to_high_price": income_p_upper,
                        "High_price": [high_price],
                        "High_price_count": [dataframe.loc[dataframe["price"] >= high_price].shape[0]]})
    df.insert (loc=0, column="Method", value=[method_name])
    return df

# 1) Bütün kategorilerin fiyatları baz alınarak tüm dataya uygulanan yaklaşım
lower, upper = DescrStatsW (df["price"]).tconfint_mean ()
median_all = df["price"].median()

all_df = price_df (df, lower, upper, median_all, "According_to_all_cat_ids")

# 2) 326584 category_id'li fiyatlar dışarıda bırakılarak hesaplanan fiyatların tüm dataya uygulandığı yaklaşım
low_lim, high_lim = DescrStatsW (df.loc[df["category_id"] != 326584, "price"]).tconfint_mean ()
median_exc32 = df.loc[df["category_id"] != 326584, "price"].median()

exc_cat3_df = price_df (df, low_lim, high_lim, median_exc32, "Excepted_326584_id_prices")

# 3) 489756 category_id'li fiyatlar dışarıda bırakılarak hesaplanan fiyatların tüm dataya uygulandığı yaklaşım
low_limit, high_limit = DescrStatsW (df.loc[df["category_id"] != 489756, "price"]).tconfint_mean ()
median_exc48 = df.loc[df["category_id"] != 489756, "price"].median()
exc_cat4_df = price_df (df, low_limit, high_limit, median_exc48, "Excepted_489756_id_prices")

# 4) Ortalama fiyatları benzer kategorilerin fiyatları baz alınarak hesaplanan fiyatların tüm dataya uygulandığı yaklaşım
lower_lim, higher_lim = DescrStatsW (df.loc[(df["category_id"] != 489756) & (df["category_id"] != 326584), "price"]).tconfint_mean ()

median_similar = df.loc[(df["category_id"] != 489756) & (df["category_id"] != 326584), "price"].median()
similar_prices_df = price_df (df, lower_lim, higher_lim, median_similar, "Similar_cat_id_prices")



final_df = pd.concat([all_df, exc_cat3_df, exc_cat4_df, similar_prices_df]).reset_index(drop=True)




# 3. Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi için karar destek sistemi oluşturunuz.

final_df        # İlgili tabloda güven aralıkları "Lower_price" ve "High_price" olarak belirtilmiştir.Güven aralıkları baz alınarak hesaplanan gelirler de "Income_according_to_low_price" ve "Income_according_to_high_price" olarak belirtilmiştir. İlgili fiyatlar aralığındaki gelir değişimi görülebilir.

sns.scatterplot(x="Lower_price", y="Income_according_to_low_price", hue="Method", data=final_df)
plt.show();

# 4. Olası fiyat değişiklikleri için item satın almalarını ve gelirlerini simüle ediniz.


income = pd.concat([final_df.iloc[:,1], final_df.iloc[:,4], final_df.iloc[:,7], final_df.iloc[:,10]])
prices = pd.concat([final_df.iloc[:,2], final_df.iloc[:,5], final_df.iloc[:,8], final_df.iloc[:,11]])

count_ = pd.concat([final_df.iloc[:,3],final_df.iloc[:,6],final_df.iloc[:,9],final_df.iloc[:,12]])

new_df = pd.concat([income,prices, count_], axis=1).reset_index(drop=True)
new_df.columns=["Income","Price","Count"]

sns.lmplot(x="Price", y="Income", data=new_df)
plt.show();


