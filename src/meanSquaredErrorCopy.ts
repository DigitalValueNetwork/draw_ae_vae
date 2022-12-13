/* import tf, { squaredDifference } from "@tensorflow/tfjs-node"
import {convertToTensor} from "@tensorflow/tfjs/src/tensor_util_env"



export function meanSquaredError_<T extends tf.Tensor, O extends tf.Tensor>(
    labels: T|tf.TensorLike, predictions: T|tf.TensorLike,
    weights?: tf.Tensor|tf.TensorLike,
    reduction = tf.Reduction.SUM_BY_NONZERO_WEIGHTS): O {
  const $labels = tf.backend_util.convert .convertToTensor(labels, 'labels', 'meanSquaredError');
  const $predictions =
      convertToTensor(predictions, 'predictions', 'meanSquaredError');
  let $weights: tf.Tensor = <any>null;
  if (weights != null) {
    $weights = convertToTensor(weights, 'weights', 'meanSquaredError');
  }

  const losses = squaredDifference($labels, $predictions);
  return tf.losses.computeWeightedLoss(losses, $weights, reduction);
}
*/

export {}
