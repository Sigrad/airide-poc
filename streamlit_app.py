                            st.markdown("**Metriken**"); mae = abs(live_df['Ist'] - live_df['Prognose_Schnitt']).mean(); st.metric("MAE", f"{mae:.2f} min"); st.markdown("**Modell-Abweichungen**"); live_df['Abs_Fehler'] = abs(live_df['Ist'] - live_df['Random Forest']); st.dataframe(live_df[['Ist', 'Random Forest', 'Abs_Fehler']].sort_values('Abs_Fehler', ascending=False).head(5), use_container_width=True)

    # TAB 4: VALIDIERUNG (GEÄNDERT)
    with tab4:
        if 'benchmark' in st.session_state:
            res = st.session_state['benchmark']
            
            # --- 1. PERFORMANCE-METRIKEN ---
            st.subheader("Performance-Metriken")
            best_model_name = min(res, key=lambda k: res[k]['rmse'])
            cols = st.columns(len(res))
            
            for idx, (name, metrics) in enumerate(res.items()):
                with cols[idx]:
                    if name == best_model_name:
                        st.success(f"{name}")
                        st.metric(label="RMSE (Fehler)", value=f"{metrics['rmse']:.2f} min", delta="Minimum", delta_color="inverse")
                        st.metric(label="R² (Erklärungskraft)", value=f"{metrics['r2']:.2f}", delta="Maximum")
                    else:
                        st.info(f"{name}")
                        st.metric(label="RMSE (Fehler)", value=f"{metrics['rmse']:.2f} min")
                        st.metric(label="R² (Erklärungskraft)", value=f"{metrics['r2']:.2f}")

            st.divider()

            # --- 2. GRAFIKEN NEBENEINANDER ---
            g_col1, g_col2 = st.columns(2)
            
            first_key = list(res.keys())[0]
            limit_line = 100 
            actuals = res[first_key]['actuals']
            
            with g_col1:
                st.subheader("Zeitreihen-Validierung")
                df_plot = pd.DataFrame({'Ist': actuals[:limit_line]})
                for name, metrics in res.items():
                    df_plot[name] = metrics['predictions'][:limit_line]
                
                fig_line, ax_line = plt.subplots(figsize=(6, 4))