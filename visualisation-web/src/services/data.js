import { request } from './request'


const queryperception = ({ model, dataset, classes, id, td, epoch }) => {
    const path = 'http://39.99.241.32:8888/post'
    const params = { model, dataset, classes, id, td, epoch }
    return request({ path, params })
}
const home = ({range}) => {
    const path = `http://39.99.241.32:8888/home`
    const params = {range}
    return request({ path, params, method: 'GET' })
}
export { queryperception, home }
