监控和告警
=================

监控
-----------------

InternEvo 使用 ``internlm.monitor.initialize_monitor_manager()`` 来初始化上下文监控管理。其中，一个实例化的单例对象 ``internlm.monitor.monitor.MonitorManager`` 将管理监控线程并使用 ``internlm.monitor.monitor.MonitorTracker`` 来跟踪模型训练生命周期和训练状态。

.. autofunction:: internlm.monitor.initialize_monitor_manager

.. autoclass:: internlm.monitor.monitor.MonitorManager
    :members:

.. autoclass:: internlm.monitor.monitor.MonitorTracker
    :members:

告警
-----------------

InternEvo 监控线程会周期性地检查模型训练过程中是否出现 loss spike、潜在的 training stuck、运行时异常等，并捕获 SIGTERM 异常信号。当出现上述情况时，将触发警报，并通过调用 ``internlm.monitor.alert.send_feishu_msg_with_webhook()`` 向飞书的 Webhook 地址发送报警消息。

.. autofunction:: internlm.monitor.alert.send_feishu_msg_with_webhook

监控告警配置
-----------------


配置由配置文件中的 ``monitor`` 字段指定， 用户可以通过修改配置文件 `config file <https://github.com/InternLM/InternEvo/blob/develop/configs/7B_sft.py>`_ 来更改监控配置。以下是一个监控配置的示例：

.. code-block:: python

    monitor = dict(
        alert=dict(
            enable_feishu_alert=False,
            feishu_alert_address=None,
            alert_file_path=f"llm_alter/{JOB_NAME}_alert.log",
        ),
    )

- enable_feishu_alert (bool)：是否启用飞书告警。默认值：False。
- feishu_alert_address (str)：飞书告警的 Webhook 地址。默认值：None。
- alert_file_path (str)：告警存储路径。默认值：None。
