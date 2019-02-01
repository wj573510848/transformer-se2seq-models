# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
#encoder
import logging

def single_cell(hidden_size,keep_prob,residual_connection,is_training,residual_fn=None):
    cell=tf.nn.rnn_cell.LSTMCell(hidden_size)
    if is_training:
        cell=tf.nn.rnn_cell.DropoutWrapper(cell,keep_prob)
    if residual_connection:
        cell=tf.nn.rnn_cell.ResidualWrapper(cell,residual_fn=residual_fn)
    return cell
def build_bidirectional_rnn(hidden_size,keep_prob,is_training,inputs, sequence_length,dtype):
    """Create and call biddirectional RNN cells.
    Args:
      inputs:batch_size,max_length,embedding_size
      num_residual_layers: Number of residual layers from top to bottom. For
        example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2 RNN
        layers in each RNN cell will be wrapped with `ResidualWrapper`..
    Returns:
      The concatenated bidirectional output and the bidirectional RNN cell"s
      state.
    """
    # Construct forward and backward cells
    fw_cell = single_cell(hidden_size,keep_prob,residual_connection=False,is_training=is_training)
    bw_cell = single_cell(hidden_size,keep_prob,residual_connection=False,is_training=is_training)

    bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
        fw_cell,
        bw_cell,
        inputs,
        dtype=dtype,
        sequence_length=sequence_length)

    return tf.concat(bi_outputs, -1), bi_state #batch_size,max_length,2*hidden_size
def build_multi_cells(num_layers,hidden_size,keep_prob,is_training,residual_fn=None):
    cells=[]
    for i in range(num_layers):
        if i==0:
            cells.append(single_cell(hidden_size,keep_prob,False,is_training))
        else:
            cells.append(single_cell(hidden_size,keep_prob,True,is_training,residual_fn=residual_fn))
    return cells
#gnmt, we build 1 bi-directional layer, and (num_encoder_layers - 1) uni-directional layers
def encoder_cells(hidden_size,keep_prob,is_training,inputs,sequence_length,num_encoder_layers,logger):
    num_bi_layers = 1
    num_uni_layers = num_encoder_layers - num_bi_layers
    logger.info("# Build a GNMT encoder")
    logger.info("  num_bi_layers = %d" % num_bi_layers)
    logger.info("  num_uni_layers = %d" % num_uni_layers)
    bi_encoder_outputs, bi_encoder_state = build_bidirectional_rnn(
            hidden_size,keep_prob,is_training,inputs, sequence_length,dtype=tf.float32)
    if num_uni_layers>0:
        cells=build_multi_cells(num_uni_layers,hidden_size,keep_prob,is_training)
        #batch_size,max_length,hidden_size
        if len(cells)>1:
            cells=tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            cells=cells[0]

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cells,
                bi_encoder_outputs,
                dtype=tf.float32,
                sequence_length=sequence_length)
    else:
        logger.error("gnmt model need more than 2 layers")
        exit(1)
    return encoder_outputs,encoder_state
def prepare_beam_search_decoder_inputs(
      beam_width, memory, source_sequence_length, encoder_state,batch_size):
    memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(
        source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    batch_size = batch_size * beam_width
    return memory, source_sequence_length, encoder_state, batch_size
def decoder_cells(encoder_outputs,encoder_state,encoder_seq_length,num_decoder_layers,is_training,hidden_size,
                  beam_width,length_penalty_weight,
                  batch_size,decoder_inputs,decoder_seq_length,target_vocab_size,tgt_sos_id,tgt_eos_id,
                  embedding_decoder,maximum_iterations,keep_prob):
    if not is_training:
        encoder_outputs, encoder_seq_length, encoder_state, batch_size =prepare_beam_search_decoder_inputs(
              beam_width, encoder_outputs, encoder_seq_length,
              encoder_state,batch_size)
    else:
      batch_size = batch_size
    cells=build_multi_cells(num_decoder_layers,hidden_size,keep_prob,is_training,residual_fn=gnmt_residual_fn)
    # Only wrap the bottom layer with the attention mechanism.
    attention_cell = cells.pop(0)
    attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        hidden_size, encoder_outputs, memory_sequence_length=encoder_seq_length)
    attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        attention_cell,
        attention_mechanism,
        attention_layer_size=None,  # don't use attention layer.
        output_attention=False,
        alignment_history=False,
        name="attention")
    cell = GNMTAttentionMultiCell(
          attention_cell, cells)
    decoder_initial_state = tuple(
          zs.clone(cell_state=es)
          if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
          for zs, es in zip(
              cell.zero_state(batch_size, tf.float32), encoder_state))
    #decoder_initial_state = cell.zero_state(batch_size, dtype)
    output_layer=tf.layers.Dense(target_vocab_size)
    if is_training:
        helper = tf.contrib.seq2seq.TrainingHelper(
            decoder_inputs, decoder_seq_length,
            )
        my_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell,
            helper,
            decoder_initial_state,)
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            swap_memory=True)
        logits = output_layer(outputs.rnn_output)
        return logits,outputs.rnn_output,output_layer
    else:
        start_tokens = tf.fill([int(batch_size/beam_width)], tgt_sos_id)
        end_token = tgt_eos_id
        my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=output_layer,
              length_penalty_weight=length_penalty_weight)
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=maximum_iterations,
            swap_memory=True)
        return outputs
class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=False):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]
          if self.use_new_attention:
            cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          else:
            cur_inp = tf.concat([cur_inp, attention_state.attention], -1)
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)
    return cur_inp, tuple(new_states)
def gnmt_residual_fn(inputs, outputs):
  """Residual function that handles different inputs and outputs inner dims.

  Args:
    inputs: cell inputs, this is actual inputs concatenated with the attention
      vector.
    outputs: cell outputs

  Returns:
    outputs + actual inputs
  """
  def split_input(inp, out):
    out_dim = out.get_shape().as_list()[-1]
    inp_dim = inp.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
  actual_inputs, _ = tf.contrib.framework.nest.map_structure(
      split_input, inputs, outputs)
  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())
  tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
  tf.contrib.framework.nest.map_structure(
      assert_shape_match, actual_inputs, outputs)
  return tf.contrib.framework.nest.map_structure(
      lambda inp, out: inp + out, actual_inputs, outputs)
if __name__=="__main__":
    hidden_size=64
    keep_prob=0.5
    is_training=True
    inputs=tf.placeholder(tf.float32,[1,10,16])
    sequence_length=tf.placeholder(tf.int32,[1])
    num_encoder_layers=5
    num_decoder_layers=4
    beam_width=4
    batch_size=1
    target_vocab_size=32
    decoder_inputs=tf.placeholder(tf.float32,[1,10,16])
    decoder_seq_length=tf.placeholder(tf.int32,[1])
    dtype=tf.float32
    a,b=encoder_cells(hidden_size,keep_prob,is_training,inputs,sequence_length,num_encoder_layers,logging)
    logits=decoder_cells(a,b,sequence_length,num_decoder_layers,is_training,hidden_size,
                  beam_width,batch_size,decoder_inputs,decoder_seq_length,target_vocab_size)
    print(logits)