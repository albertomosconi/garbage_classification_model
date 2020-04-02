# Classification of garbage photos into 6 categories  
Useful imports:


```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os # file handling
import cv2 #images

import matplotlib.pyplot as plt # graphs
%matplotlib inline 

from tqdm import tqdm # progress bar
```

## Reading files
The first step is to cycle the folders containing the images and saving them in the respective lists.  
One useful thing to do here is to resize the files in order to make them easier to work with.


```python
trash = []
cardboard = []
plastic = []
metal = []
paper = []
glass = []
garbage = []

prova = []
for dirname, _, filenames in os.walk('D:/PROJECTS/AI projects/garbage-classification/Garbage classification'):
#     print(dirname)
    for filename in filenames:
        prova.append(os.path.join(dirname, filename))
# print(prova)
height = 192
width = 256

dim = (width, height)


for file in tqdm(prova):
    image = cv2.imread(file,0)
    res = cv2.resize(image, dim, interpolation = cv2.INTER_LINEAR)
    file = file.split("\\")
    
    cat = file[-2]
    if cat == 'trash':
        trash.append(res)
    if cat == 'plastic':
        plastic.append(res)
    if cat == 'glass':
        glass.append(res)
    if cat == 'paper':
        paper.append(res)
    if cat == 'metal':
        metal.append(res)
    if cat == 'cardboard':
        cardboard.append(res)
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2527/2527 [00:01<00:00, 1348.28it/s]
    

Once we have the 6 lists we create the merged dataset joining them, and we create the target columns containing the correct label for each of the images.


```python
garbage = trash + plastic + glass + paper + metal + cardboard
target = ['trash' for item in trash] + ['plastic' for item in plastic]+['glass' for item in glass]+['paper' for item in paper]+['metal' for item in metal]+['cardboard' for item in cardboard]
print("Total images:",len(garbage))
print("Total labels:",len(target))
```

    Total images: 2527
    Total labels: 2527
    

We can now preview one of the images to see what we're working with.


```python
plt.imshow(garbage[0], cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```


![png](index_files/index_7_0.png)


## Input manipulation
The next step is to prepare the images to be fed to the prediction model, this means taking each one, that is currently represented by a matrix of pixels, and "flattening" it to an array of pixels. Each pixel contains its grayscale value.


```python
for i in range(len(garbage)):
    garbage[i] = garbage[i].reshape(width*height)

```


```python
print("tot foto:",len(garbage))
print(garbage[0])
```

    tot foto: 2527
    [219 219 219 ...  80  80  79]
    

We join the images with the target column into a pandas DataFrame object.


```python
df = pd.DataFrame(np.matrix(garbage))
df['target'] = target
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>49143</th>
      <th>49144</th>
      <th>49145</th>
      <th>49146</th>
      <th>49147</th>
      <th>49148</th>
      <th>49149</th>
      <th>49150</th>
      <th>49151</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>219</td>
      <td>221</td>
      <td>221</td>
      <td>...</td>
      <td>83</td>
      <td>83</td>
      <td>82</td>
      <td>82</td>
      <td>81</td>
      <td>81</td>
      <td>80</td>
      <td>80</td>
      <td>79</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>1</th>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>203</td>
      <td>...</td>
      <td>75</td>
      <td>73</td>
      <td>74</td>
      <td>75</td>
      <td>75</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>74</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>2</th>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>...</td>
      <td>83</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>3</th>
      <td>211</td>
      <td>212</td>
      <td>213</td>
      <td>213</td>
      <td>214</td>
      <td>214</td>
      <td>213</td>
      <td>212</td>
      <td>212</td>
      <td>212</td>
      <td>...</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>76</td>
      <td>75</td>
      <td>74</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>4</th>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>221</td>
      <td>222</td>
      <td>...</td>
      <td>36</td>
      <td>42</td>
      <td>59</td>
      <td>66</td>
      <td>63</td>
      <td>56</td>
      <td>55</td>
      <td>48</td>
      <td>39</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2522</th>
      <td>230</td>
      <td>236</td>
      <td>235</td>
      <td>236</td>
      <td>234</td>
      <td>234</td>
      <td>233</td>
      <td>232</td>
      <td>233</td>
      <td>233</td>
      <td>...</td>
      <td>176</td>
      <td>166</td>
      <td>165</td>
      <td>172</td>
      <td>171</td>
      <td>172</td>
      <td>183</td>
      <td>179</td>
      <td>174</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>2523</th>
      <td>188</td>
      <td>188</td>
      <td>188</td>
      <td>188</td>
      <td>187</td>
      <td>189</td>
      <td>188</td>
      <td>185</td>
      <td>182</td>
      <td>181</td>
      <td>...</td>
      <td>159</td>
      <td>159</td>
      <td>160</td>
      <td>161</td>
      <td>161</td>
      <td>165</td>
      <td>165</td>
      <td>172</td>
      <td>165</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>2524</th>
      <td>150</td>
      <td>147</td>
      <td>149</td>
      <td>147</td>
      <td>146</td>
      <td>148</td>
      <td>150</td>
      <td>150</td>
      <td>147</td>
      <td>148</td>
      <td>...</td>
      <td>94</td>
      <td>91</td>
      <td>77</td>
      <td>76</td>
      <td>74</td>
      <td>74</td>
      <td>76</td>
      <td>84</td>
      <td>91</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>2525</th>
      <td>188</td>
      <td>189</td>
      <td>187</td>
      <td>186</td>
      <td>185</td>
      <td>185</td>
      <td>185</td>
      <td>185</td>
      <td>183</td>
      <td>183</td>
      <td>...</td>
      <td>164</td>
      <td>155</td>
      <td>151</td>
      <td>139</td>
      <td>140</td>
      <td>142</td>
      <td>141</td>
      <td>134</td>
      <td>138</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>2526</th>
      <td>227</td>
      <td>226</td>
      <td>226</td>
      <td>225</td>
      <td>224</td>
      <td>224</td>
      <td>223</td>
      <td>222</td>
      <td>223</td>
      <td>223</td>
      <td>...</td>
      <td>207</td>
      <td>209</td>
      <td>209</td>
      <td>209</td>
      <td>209</td>
      <td>208</td>
      <td>208</td>
      <td>208</td>
      <td>208</td>
      <td>cardboard</td>
    </tr>
  </tbody>
</table>
<p>2527 rows × 49153 columns</p>
</div>



## Shuffling the data
It is important to shuffle the input data to prevent our classfier to be trained incorrectly.


```python
from sklearn.utils import shuffle
df = shuffle(df)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>49143</th>
      <th>49144</th>
      <th>49145</th>
      <th>49146</th>
      <th>49147</th>
      <th>49148</th>
      <th>49149</th>
      <th>49150</th>
      <th>49151</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66</th>
      <td>232</td>
      <td>232</td>
      <td>231</td>
      <td>230</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>228</td>
      <td>227</td>
      <td>...</td>
      <td>128</td>
      <td>128</td>
      <td>129</td>
      <td>130</td>
      <td>130</td>
      <td>130</td>
      <td>131</td>
      <td>131</td>
      <td>130</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>1782</th>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>201</td>
      <td>202</td>
      <td>202</td>
      <td>202</td>
      <td>202</td>
      <td>201</td>
      <td>201</td>
      <td>...</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>221</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>222</td>
      <td>metal</td>
    </tr>
    <tr>
      <th>943</th>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>227</td>
      <td>226</td>
      <td>226</td>
      <td>...</td>
      <td>168</td>
      <td>167</td>
      <td>168</td>
      <td>168</td>
      <td>169</td>
      <td>168</td>
      <td>168</td>
      <td>168</td>
      <td>168</td>
      <td>glass</td>
    </tr>
    <tr>
      <th>2279</th>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>255</td>
      <td>254</td>
      <td>254</td>
      <td>254</td>
      <td>254</td>
      <td>254</td>
      <td>254</td>
      <td>...</td>
      <td>138</td>
      <td>177</td>
      <td>177</td>
      <td>177</td>
      <td>177</td>
      <td>176</td>
      <td>175</td>
      <td>174</td>
      <td>172</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>597</th>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>205</td>
      <td>...</td>
      <td>137</td>
      <td>139</td>
      <td>138</td>
      <td>138</td>
      <td>137</td>
      <td>137</td>
      <td>138</td>
      <td>137</td>
      <td>135</td>
      <td>plastic</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2393</th>
      <td>231</td>
      <td>233</td>
      <td>234</td>
      <td>234</td>
      <td>234</td>
      <td>232</td>
      <td>226</td>
      <td>225</td>
      <td>223</td>
      <td>229</td>
      <td>...</td>
      <td>81</td>
      <td>81</td>
      <td>82</td>
      <td>82</td>
      <td>81</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>78</td>
      <td>cardboard</td>
    </tr>
    <tr>
      <th>658</th>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>224</td>
      <td>...</td>
      <td>47</td>
      <td>46</td>
      <td>44</td>
      <td>41</td>
      <td>40</td>
      <td>39</td>
      <td>47</td>
      <td>50</td>
      <td>35</td>
      <td>glass</td>
    </tr>
    <tr>
      <th>109</th>
      <td>230</td>
      <td>230</td>
      <td>230</td>
      <td>230</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>229</td>
      <td>...</td>
      <td>104</td>
      <td>104</td>
      <td>103</td>
      <td>103</td>
      <td>102</td>
      <td>101</td>
      <td>100</td>
      <td>100</td>
      <td>99</td>
      <td>trash</td>
    </tr>
    <tr>
      <th>1862</th>
      <td>195</td>
      <td>195</td>
      <td>195</td>
      <td>195</td>
      <td>194</td>
      <td>194</td>
      <td>194</td>
      <td>194</td>
      <td>193</td>
      <td>193</td>
      <td>...</td>
      <td>29</td>
      <td>95</td>
      <td>157</td>
      <td>159</td>
      <td>157</td>
      <td>157</td>
      <td>156</td>
      <td>156</td>
      <td>155</td>
      <td>metal</td>
    </tr>
    <tr>
      <th>790</th>
      <td>179</td>
      <td>179</td>
      <td>179</td>
      <td>179</td>
      <td>181</td>
      <td>180</td>
      <td>180</td>
      <td>181</td>
      <td>181</td>
      <td>181</td>
      <td>...</td>
      <td>194</td>
      <td>195</td>
      <td>195</td>
      <td>195</td>
      <td>195</td>
      <td>194</td>
      <td>194</td>
      <td>194</td>
      <td>194</td>
      <td>glass</td>
    </tr>
  </tbody>
</table>
<p>2527 rows × 49153 columns</p>
</div>



## Splitting the dataset
We split the dataset into two sections, one that we'll use for training the model and the other for testing it's accuracy. Same thing is done with the 'target' column.


```python
y = df["target"]
df.drop("target",1, inplace=True)
```

    c:\users\alberto\appdata\local\programs\python\python38\lib\site-packages\pandas\core\frame.py:3990: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      return super().drop(
    
