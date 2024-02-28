import axios from 'axios'
import router from '../router/index'
const handleErrorCode = ({ code }) => {
  switch (String(code)) {
    case '400':
      break
    case '101':
      router.push({
        path: `/`
      })
      break
    default:
      break
  }
}

// 构造 FormData 对象数据
const generateFormData = (data) => {
  return Object.keys(data).reduce((previous, current) => {
    const currentFormValue = data[current]
    if (Array.isArray(currentFormValue)) {
      currentFormValue.map((currentFormItem) => previous.append(current, currentFormItem))
    } else {
      previous.append(current, currentFormValue)
    }

    return previous
  }, new FormData())
}

/**
 *
 * @param {String} path 接口路径 '/xxx/xxx'
 * @param {String} method 请求方法
 *
 * 获取相应功能模块使用的 baseURL，目前仅包含以下四种值
 * 通用模块：'common'
 * 项目模块：'project'
 * 埋点模块：'buriedPoint'
 * 高德 Web 服务：'amap'
 * @param {String} baseURLSign
 *
 * @param {Boolean} hasAuthHeader 是否把 token 设置在 Headers 中发送
 *
 * 以哪种类型提交数据，此参数仅当 method 等于 'POST' 时有效
 * 以 multipart/form-data 提交数据：'formData'
 * 以 application/json 提交数据: 'json'
 * 备注：文件类型的提交，不需要提前添加到 FormData 对象中
 * @param {String} contentType
 *
 * @param {Object} params 提交的数据
 *
 */

// 请求拦截器
axios.interceptors.request.use(
  (config) => {
    // config.headers['Access-Control-Max-Age'] = 86400
    return config
  },
  (error) => {
    Promise.reject(error)
  }
)

const request = ({
  path,
  method = 'POST',
  // baseURLSign = 'common',
  hasAuthHeader = false,
  contentType = 'formData',
  params = {}
}) => {
  let requestConfig = {
    url: path,
    method,
    // baseURL: currentBaseURL,
    baseURL: '/api',
    timeout: 10000
  }

  if (method === 'GET') {
    if (hasAuthHeader) {
      requestConfig = {
        ...requestConfig
      }
    }

    requestConfig = { ...requestConfig, params }
  } else if (method === 'POST') {
    if (contentType === 'formData') {
      params = generateFormData(params)

      if (hasAuthHeader) {
        requestConfig = {
          ...requestConfig
        }
      } else {
        // params.append('token', token)
      }
    } else if (contentType === 'json') {
        requestConfig = {
            ...requestConfig,
            headers: {
              'Content-Type': 'application/json'
            }
          }
          params = JSON.stringify({
            ...params
          })
    } else {
      params = { ...params }
    }

    requestConfig = {
      ...requestConfig,
      data: params
    }
  }

  return axios(requestConfig)
    .then((result) => {
      const { data } = result
      const { code, content, message } = data
      if (!content) {
        handleErrorCode({ code, content, message })
      }
      return data
    })
    .catch((error) => {
      console.error('get data error: ', error)
    })
}

export { request }
